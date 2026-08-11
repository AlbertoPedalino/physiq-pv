"""SDE-Net training loop for the PVGIS ST-GNN run.

Implements Algorithm 1 of Kong et al. (2020): the drift net f (and the encoder /
GAT / heads) is trained on the in-distribution prediction loss, while the
diffusion net g is trained alternately as a binary discriminator: 0 for an
in-distribution latent and 1 for a Gaussian-noise pseudo-OOD latent. The two
optimisers are SGD with the paper's momentum and weight decay. The prediction
path uses one Brownian trajectory per update; uncertainty is read off from
multiple paths at inference (see training/uncertainty.py).

As in the authors' regression experiment, the prediction loss is the
heteroscedastic Gaussian NLL over the PV head's (mean, sigma) output (aleatoric
uncertainty); the optional irradiance head keeps a plain MSE.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from physiq_pv.data.pvgis_dataset import PVGISWindowDataset
from physiq_pv.model.st_gnn import STGNN
from physiq_pv.model.sde_net import diffusion_bce_loss
from physiq_pv.training.losses import gaussian_nll, make_loss_fn


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
    use_irradiance_loss: bool = True,
    irradiance_loss_weight: float = 1.0,
    ood_noise_std: float = 2.0,
    lr_g: Optional[float] = 0.01,
    feature_names: Optional[List[str]] = None,
    train_normal_only: bool = False,
    sde_sigma_initial: float = 0.01,
    sde_sigma_warmup_epochs: int = 30,
    gradient_clip_norm: float = 100.0,
    lr_decay_epoch: int = 20,
    lr_decay_factor: float = 0.1,
    validation_metric: str = "rmse_daytime",
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
) -> STGNN:
    """Train one SDE-Net ST-GNN (alternating drift / diffusion optimisation).

    The diffusion objective uses the paper's pseudo-OOD construction:
    ``x_ood = x + epsilon``, ``epsilon ~ N(0, ood_noise_std^2 I)`` over every
    input channel. The v1 paper's YearMSD schedule uses ``sigma=0.01`` for the
    first 30 epochs then the model's configured final sigma (normally 0.5);
    the public repo uses ``0.1`` for the initial value.
    The public YearMSD optimiser clips the prediction gradients to norm 100
    and multiplies only the drift/backbone/head learning rate by 0.1 after
    zero-indexed epoch 20; the diffusion learning rate remains unchanged.
    Per-epoch metrics are stored on ``model.train_loss_history``.
    """
    if validation_dataset is None:
        raise ValueError("A disjoint validation_dataset is required.")
    if validation_metric not in {"rmse_daytime", "mae_daytime", "nll"}:
        raise ValueError(
            "validation_metric must be one of rmse_daytime, mae_daytime, nll."
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
    if not np.isfinite(sde_sigma_initial) or sde_sigma_initial <= 0.0:
        raise ValueError(
            "sde_sigma_initial must be finite and > 0; "
            f"got {sde_sigma_initial}."
        )
    if sde_sigma_warmup_epochs < 0:
        raise ValueError(
            "sde_sigma_warmup_epochs must be >= 0; "
            f"got {sde_sigma_warmup_epochs}."
        )
    if not np.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
        raise ValueError(
            "gradient_clip_norm must be finite and > 0; "
            f"got {gradient_clip_norm}."
        )
    if lr_decay_epoch < 0:
        raise ValueError(f"lr_decay_epoch must be >= 0; got {lr_decay_epoch}.")
    if not np.isfinite(lr_decay_factor) or not 0.0 < lr_decay_factor <= 1.0:
        raise ValueError(
            "lr_decay_factor must be finite and in (0, 1]; "
            f"got {lr_decay_factor}."
        )

    # Retained for caller compatibility.  The faithful SDE-Net pseudo-OOD
    # transformation perturbs the complete input, rather than a hand-picked
    # feature subset.
    del feature_names

    keep_all = None
    if train_normal_only:
        if not getattr(dataset, "event_filter_applied", False):
            raise ValueError(
                "train_normal_only=True requires a dataset physically filtered "
                "with regional event labels."
            )
        if not getattr(validation_dataset, "event_filter_applied", False):
            raise ValueError(
                "train_normal_only=True requires an event-filtered validation dataset."
            )
        if (
            dataset.event_rare_target_all.any()
            or dataset.event_rare_history_all.any()
        ):
            raise ValueError(
                "A rare event window survived the training filter."
            )
        if (
            validation_dataset.event_rare_target_all.any()
            or validation_dataset.event_rare_history_all.any()
        ):
            raise ValueError(
                "A rare event window survived the validation filter."
            )
        print(
            "  [stgnn] train-normal-only: datasets contain only graph-wide "
            f"normal events (train={len(dataset)}, "
            f"validation={len(validation_dataset)})"
        )

    kt_max = float(getattr(model, "kt_poa_max", 1.6))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device)
    sde_sigma_final = float(model.sde.sigma)
    if sde_sigma_initial > sde_sigma_final:
        raise ValueError(
            "sde_sigma_initial must not exceed the configured final sigma; "
            f"got {sde_sigma_initial} > {sde_sigma_final}."
        )

    # Two optimisers (Algorithm 1): opt_f over the drift + encoder + GAT + heads,
    # opt_g over the diffusion net only.  These are the SGD settings used in the
    # authors' MNIST, SVHN and YearMSD scripts.
    g_params = list(model.sde.diffusion_net.parameters())
    g_ids = {id(p) for p in g_params}
    f_params = [p for p in model.parameters() if id(p) not in g_ids]
    opt_f = torch.optim.SGD(f_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    opt_g = torch.optim.SGD(
        g_params, lr=lr if lr_g is None else lr_g, momentum=0.9, weight_decay=5e-4
    )

    # Per-element auxiliary MSE (reduction="none") for the irradiance head;
    # _masked_mean collapses to a plain mean when keep is None. The PV head uses
    # the Gaussian NLL (gaussian_nll).
    loss_fn = make_loss_fn(reduction="none")
    loss_label = "nll"

    def _masked_mean(loss_elem, keep):
        """Mean over kept cells; direct horizons share each node mask."""
        if keep is None:
            return loss_elem.mean()
        while keep.ndim < loss_elem.ndim:
            keep = keep.unsqueeze(-1)
        keep = keep.expand_as(loss_elem)
        tot = keep.sum()
        if tot == 0:
            return (loss_elem * 0.0).sum()
        return (loss_elem * keep).sum() / tot

    def _kt_target(k):
        return torch.from_numpy(
            np.clip(dataset.kt_poa_target_all[k.numpy()], 0.0, kt_max)
        ).to(device)

    @torch.no_grad()
    def _validate() -> dict:
        model.eval()
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        daytime_count = 0
        nll_sum = 0.0
        nll_count = 0
        scale = torch.from_numpy(validation_dataset.pv_scale).to(device)
        solar_targets = validation_dataset.solar_irradiance_poa_target_all
        if solar_targets is None:
            raise ValueError(
                "Validation dataset requires physical POA targets for daytime metrics."
            )
        for x, y, k in validation_loader:
            x, y = x.to(device), y.to(device)
            _, mean, sigma = model(x, ei, ew, None, stochastic=False)
            nll_elem = gaussian_nll(y, mean, sigma)
            nll_sum += float(nll_elem.sum().item())
            nll_count += int(nll_elem.numel())

            scale_shape = (1, -1) if mean.ndim == 2 else (1, -1, 1)
            pred_raw = mean * scale.view(*scale_shape)
            true_raw = torch.from_numpy(
                validation_dataset.y_true_all[k.numpy()]
            ).to(device)
            daylight = torch.from_numpy(
                solar_targets[k.numpy()] >= 10.0
            ).to(device)
            error = pred_raw - true_raw
            squared_error_sum += float((error[daylight] ** 2).sum().item())
            absolute_error_sum += float(error[daylight].abs().sum().item())
            daytime_count += int(daylight.sum().item())
        if daytime_count == 0:
            raise ValueError("Validation split contains no daytime target cells.")
        return {
            "validation/rmse_daytime": float(
                np.sqrt(squared_error_sum / daytime_count)
            ),
            "validation/mae_daytime": float(
                absolute_error_sum / daytime_count
            ),
            "validation/nll": float(nll_sum / nll_count),
        }

    history: List[dict] = []
    best_score = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    t_train = time.perf_counter()
    for ep in range(epochs):
        model.train()
        model.sde.sigma = (
            float(sde_sigma_initial)
            if ep < sde_sigma_warmup_epochs
            else sde_sigma_final
        )
        t_ep = time.perf_counter()
        losses, losses_pv, losses_irr = [], [], []
        g_in_list, g_ood_list = [], []
        losses_g, losses_g_in, losses_g_ood = [], [], []
        for x, y, k in loader:
            x, y = x.to(device), y.to(device)
            keep = (
                torch.from_numpy(keep_all[k.numpy()]).to(device)
                if keep_all is not None else None
            )

            # --- drift step: Gaussian NLL PV loss on the in-distribution
            # prediction (aleatoric head). Under train_normal_only the dataset
            # has already been physically filtered.
            pred_poa, pred_pv_mean, pred_pv_sigma = model(
                x, ei, ew, None, stochastic=True
            )
            loss_pv = _masked_mean(gaussian_nll(y, pred_pv_mean, pred_pv_sigma), keep)
            loss = loss_pv
            if use_irradiance_loss:
                loss_irr = _masked_mean(loss_fn(pred_poa, _kt_target(k)), keep)
                loss = loss + irradiance_loss_weight * loss_irr
                losses_irr.append(float(loss_irr.item()))
            opt_f.zero_grad()
            loss.backward()
            # Algorithm 1 updates only h1/backbone, drift f and the output
            # heads in this step.  Restrict clipping to the same parameter set:
            # diffusion gradients are intentionally ignored here and may still
            # be present from the preceding opt_g update.
            torch.nn.utils.clip_grad_norm_(f_params, gradient_clip_norm)
            opt_f.step()

            # --- diffusion step: BCE(ID=0, pseudo-OOD=1), as in Algorithm 1 ---
            # The encoder is detached here just as the public SDE-Net code calls
            # diffusion(out.detach()): only g is updated in this step.
            x_ood = x + float(ood_noise_std) * torch.randn_like(x)
            with torch.no_grad():
                x0_in = model.encode(x, ei, ew)
                x0_ood = model.encode(x_ood, ei, ew)
            g_in = model.sde.diffusion(x0_in)
            g_ood = model.sde.diffusion(x0_ood)
            loss_g, loss_g_in, loss_g_ood = diffusion_bce_loss(g_in, g_ood)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            losses.append(float(loss.item()))
            losses_pv.append(float(loss_pv.item()))
            g_in_list.append(float(g_in.mean().item()))
            g_ood_list.append(float(g_ood.mean().item()))
            losses_g.append(float(loss_g.item()))
            losses_g_in.append(float(loss_g_in.item()))
            losses_g_ood.append(float(loss_g_ood.item()))

        g_in_m, g_ood_m = float(np.mean(g_in_list)), float(np.mean(g_ood_list))
        rec = {
            "loss/total": float(np.mean(losses)),
            "loss/pv": float(np.mean(losses_pv)),
            "train/g_in": g_in_m,
            "train/g_ood": g_ood_m,
            "train/g_ratio": float(g_ood_m / g_in_m) if g_in_m > 0.0 else float("nan"),
            "loss/diffusion": float(np.mean(losses_g)),
            "loss/diffusion_in": float(np.mean(losses_g_in)),
            "loss/diffusion_ood": float(np.mean(losses_g_ood)),
            "train/sigma": float(model.sde.sigma),
            "train/lr_f": float(opt_f.param_groups[0]["lr"]),
            "train/lr_g": float(opt_g.param_groups[0]["lr"]),
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
        # Match the public YearMSD script: decay opt_f after epoch index 20.
        # opt_g intentionally keeps its original learning rate.
        if ep == lr_decay_epoch:
            for param_group in opt_f.param_groups:
                param_group["lr"] *= float(lr_decay_factor)
            print(
                f"  [stgnn] lr_f decay after epoch index {ep}: "
                f"{rec['train/lr_f']:.6g} -> {opt_f.param_groups[0]['lr']:.6g}"
            )
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"  [stgnn] early stopping: no {validation_metric} improvement "
                f"for {early_stopping_patience} epochs."
            )
            break
    if best_state is None:
        raise RuntimeError("Training completed without a valid validation checkpoint.")
    model.load_state_dict(best_state)
    model.sde.sigma = sde_sigma_final
    model.train_loss_history = history
    model.best_epoch = best_epoch
    model.best_validation_metric = validation_metric
    model.best_validation_score = best_score
    print(f"  [stgnn] [time] train_model total: {time.perf_counter() - t_train:.1f}s")
    return model
