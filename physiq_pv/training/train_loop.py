"""SDE-Net training loop for the PVGIS ST-GNN run.

Implements the alternating training in Monaco et al.'s public SDE U-Net:
the BiLSTM/GAT drift path and heads receive the MSE point loss, while the
parallel diffusion encoder receives the sum of per-stage BCE objectives —
g_i -> 0 in-distribution and g_i -> 1 on Gaussian-noise pseudo-OOD inputs.
Uncertainty is read off at inference (see training/uncertainty.py).
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from physiq_pv.data.pvgis_dataset import PVGISWindowDataset
from physiq_pv.model.st_gnn import STGNN
from physiq_pv.training.losses import make_loss_fn
from physiq_pv.training.noise import build_noise_feature_indices, inject_input_noise


def train_model(
    model: STGNN,
    dataset: PVGISWindowDataset,
    validation_dataset: PVGISWindowDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    use_irradiance_loss: bool = False,
    irradiance_loss_weight: float = 1.0,
    ood_noise_std: float = 1.0,
    lr_g: Optional[float] = None,
    feature_names: Optional[List[str]] = None,
    train_normal_only: bool = False,
    validation_metric: str = "rmse_daytime",
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
) -> STGNN:
    """Train one SDE-Net ST-GNN (alternating drift / diffusion optimisation).

    The diffusion objective needs a pseudo-OOD batch: training inputs + Gaussian
    noise of std `ood_noise_std` (sin_elev/cos_elev excluded via feature_names).
    Per-epoch metrics are stored on `model.train_loss_history`.
    """
    if validation_dataset is None:
        raise ValueError("A disjoint validation_dataset is required.")
    if validation_metric not in {"rmse_daytime", "mae_daytime", "mse"}:
        raise ValueError(
            "validation_metric must be one of rmse_daytime, mae_daytime, mse."
        )
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0.")
    if use_irradiance_loss:
        if getattr(model, "head_poa", None) is None:
            raise ValueError(
                "use_irradiance_loss=True requires a model with an irradiance head "
                "(STGNN with use_irradiance_head=True); this model has no head_poa."
            )
        if dataset.kt_poa_target_all is None:
            raise ValueError(
                "use_irradiance_loss=True requires kt_poa targets on the training "
                "dataset (build_datasets attaches them via kt_poa_by_year)."
            )
        if not np.isfinite(irradiance_loss_weight) or irradiance_loss_weight < 0.0:
            raise ValueError(
                f"irradiance_loss_weight must be finite and >= 0, got {irradiance_loss_weight}."
            )
    if not np.isfinite(ood_noise_std) or ood_noise_std <= 0.0:
        raise ValueError(
            "ood_noise_std must be finite and > 0 (the diffusion net needs a "
            f"perturbed OOD batch to push g high); got {ood_noise_std}."
        )

    # OOD channels: every continuous channel (sin_elev/cos_elev excluded). Falls
    # back to all channels when feature names are not provided.
    if feature_names is not None:
        noise_idx = build_noise_feature_indices(feature_names)
        if not noise_idx:
            raise ValueError(
                f"no continuous feature channels for OOD noise (feature_names={feature_names})."
            )
    else:
        noise_idx = list(range(dataset[0][0].shape[-1]))
    noise_idx_t = torch.tensor(noise_idx, dtype=torch.long, device=device)

    if train_normal_only:
        if not dataset.event_filter_applied:
            raise ValueError(
                "train_normal_only=True requires a dataset physically filtered "
                "with regional event labels."
            )
        if not validation_dataset.event_filter_applied:
            raise ValueError(
                "train_normal_only=True requires an event-filtered validation dataset."
            )
        if dataset.event_rare_target_all.any() or dataset.event_rare_history_all.any():
            raise ValueError("A rare event window survived the training filter.")
        if (
            validation_dataset.event_rare_target_all.any()
            or validation_dataset.event_rare_history_all.any()
        ):
            raise ValueError("A rare event window survived the validation filter.")
        print(
            "  [stgnn] train-normal-only: datasets contain only graph-wide "
            f"normal events (train={len(dataset)}, validation={len(validation_dataset)})"
        )

    kt_max = float(getattr(model, "kt_poa_max", 1.6))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device)

    # Two optimisers as in Monaco's public SDE U-Net: opt_f over the drift
    # BiLSTM/GAT/head path, opt_g over the parallel diffusion encoder only.
    g_params = list(model.diffusion_parameters())
    g_ids = {id(p) for p in g_params}
    f_params = [p for p in model.parameters() if id(p) not in g_ids]
    opt_f = torch.optim.Adam(f_params, lr=lr)
    opt_g = torch.optim.Adam(g_params, lr=lr if lr_g is None else lr_g)

    loss_fn = make_loss_fn(reduction="mean")
    loss_label = "mse"

    def _diffusion_bce(g, target):
        """BCE for one Monaco diffusion stage over a complete retained event."""
        tgt = torch.full_like(g, float(target))
        return F.binary_cross_entropy(g, tgt, reduction="mean")

    def _kt_target(k):
        return torch.from_numpy(
            np.clip(dataset.kt_poa_target_all[k.numpy()], 0.0, kt_max)
        ).to(device)

    @torch.no_grad()
    def _validate() -> dict:
        model.eval()
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        all_squared_error_sum = 0.0
        daytime_count = 0
        all_count = 0
        scale = torch.from_numpy(validation_dataset.pv_scale).to(device)
        solar_targets = validation_dataset.solar_irradiance_poa_target_all
        if solar_targets is None:
            raise ValueError(
                "Validation dataset requires physical POA targets for daytime metrics."
            )
        for x, _, k in validation_loader:
            x = x.to(device)
            _, pred_norm = model(x, ei, ew, None, stochastic=False)
            pred_raw = pred_norm * scale.view(1, -1)
            true_raw = torch.from_numpy(
                validation_dataset.y_true_all[k.numpy()]
            ).to(device)
            daylight = torch.from_numpy(
                solar_targets[k.numpy()] >= 10.0
            ).to(device)
            error = pred_raw - true_raw
            squared_error_sum += float((error[daylight] ** 2).sum().item())
            absolute_error_sum += float(error[daylight].abs().sum().item())
            all_squared_error_sum += float((error ** 2).sum().item())
            daytime_count += int(daylight.sum().item())
            all_count += int(error.numel())
        if daytime_count == 0:
            raise ValueError("Validation split contains no daytime target cells.")
        return {
            "validation/rmse_daytime": float(
                np.sqrt(squared_error_sum / daytime_count)
            ),
            "validation/mae_daytime": float(
                absolute_error_sum / daytime_count
            ),
            "validation/mse": float(all_squared_error_sum / all_count),
        }

    history: List[dict] = []
    best_score = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    t_train = time.perf_counter()
    for ep in range(epochs):
        model.train()
        t_ep = time.perf_counter()
        losses, losses_pv, losses_irr = [], [], []
        g_in_list, g_ood_list = [], []
        for x, y, k in loader:
            x, y = x.to(device), y.to(device)

            # --- drift step: MSE PV loss on the in-distribution prediction ---
            pred_poa, pred_pv = model(x, ei, ew, None, stochastic=True)
            loss_pv = loss_fn(pred_pv, y)
            loss = loss_pv
            if use_irradiance_loss:
                loss_irr = loss_fn(pred_poa, _kt_target(k))
                loss = loss + irradiance_loss_weight * loss_irr
                losses_irr.append(float(loss_irr.item()))
            opt_f.zero_grad()
            loss.backward()
            opt_f.step()

            # --- diffusion step (Monaco/Kong): sum BCE over every aligned stage;
            #     g_i -> 0 ID and g_i -> 1 on Gaussian-noise pseudo-OOD inputs. ---
            x_ood = inject_input_noise(x, noise_idx_t, ood_noise_std, 1.0)
            g_in_terms = model.diffusion(x.detach(), ei, ew)
            g_ood_terms = model.diffusion(x_ood.detach(), ei, ew)
            loss_g = sum(_diffusion_bce(g, 0.0) for g in g_in_terms)
            loss_g = loss_g + sum(_diffusion_bce(g, 1.0) for g in g_ood_terms)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            losses.append(float(loss.item()))
            losses_pv.append(float(loss_pv.item()))
            # Log a stage-averaged raw gate for the same compact g_ratio diagnostic.
            g_in_cell = torch.stack([g.mean(-1) for g in g_in_terms]).mean(0)
            g_ood_cell = torch.stack([g.mean(-1) for g in g_ood_terms]).mean(0)
            g_in_list.append(float(g_in_cell.mean().item()))
            g_ood_list.append(float(g_ood_cell.mean().item()))

        g_in_m, g_ood_m = float(np.mean(g_in_list)), float(np.mean(g_ood_list))
        rec = {
            "loss/total": float(np.mean(losses)),
            "loss/pv": float(np.mean(losses_pv)),
            "train/g_in": g_in_m,
            "train/g_ood": g_ood_m,
            "train/g_ratio": float(g_ood_m / g_in_m) if g_in_m > 0.0 else float("nan"),
        }
        if use_irradiance_loss:
            rec["loss/irradiance"] = float(np.mean(losses_irr))
        validation = _validate()
        rec.update(validation)
        score = float(validation[f"validation/{validation_metric}"])
        improved = score < best_score - early_stopping_min_delta
        if improved:
            best_score = score
            best_epoch = ep
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        extra = (
            f"  g_in={g_in_m:.5f}  g_ood={g_ood_m:.5f}  g_ratio={rec['train/g_ratio']:.3f}"
        )
        if use_irradiance_loss:
            print(
                f"  [stgnn] epoch {ep + 1}/{epochs}  loss/total={rec['loss/total']:.5f}  "
                f"loss/pv={rec['loss/pv']:.5f}  loss/irradiance={rec['loss/irradiance']:.5f}  "
                f"(loss={loss_label}, weight={irradiance_loss_weight}){extra}  "
                f"val_{validation_metric}={score:.5f}  "
                f"[time] epoch: {time.perf_counter() - t_ep:.1f}s"
            )
        else:
            print(
                f"  [stgnn] epoch {ep + 1}/{epochs}  train_{loss_label}(norm)={np.mean(losses):.5f}"
                f"{extra}  val_{validation_metric}={score:.5f}  "
                f"[time] epoch: {time.perf_counter() - t_ep:.1f}s"
            )
        history.append(rec)
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"  [stgnn] early stopping: no {validation_metric} improvement "
                f"for {early_stopping_patience} epochs."
            )
            break
    if best_state is None:
        raise RuntimeError("Training completed without a valid validation checkpoint.")
    model.load_state_dict(best_state)
    model.train_loss_history = history
    model.best_epoch = best_epoch
    model.best_validation_metric = validation_metric
    model.best_validation_score = best_score
    print(f"  [stgnn] [time] train_model total: {time.perf_counter() - t_train:.1f}s")
    return model
