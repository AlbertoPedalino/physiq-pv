"""SDE-Net training loop for the PVGIS ST-GNN run.

Implements Algorithm 1 of Kong et al. (2020): the drift net f (and the encoder /
GAT / heads) is trained on the point loss over in-distribution data, while the
diffusion net g is trained alternately to be LOW in-distribution and HIGH on a
Gaussian-noise pseudo-OOD batch. The two share one Brownian path per step during
training; uncertainty is read off at inference (see training/uncertainty.py).
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from physiq_pv.data.pvgis_dataset import PVGISWindowDataset
from physiq_pv.model.st_gnn import STGNN
from physiq_pv.training.losses import make_loss_fn
from physiq_pv.training.noise import build_noise_feature_indices, inject_input_noise


def train_model(
    model: STGNN,
    dataset: PVGISWindowDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    use_irradiance_loss: bool = False,
    irradiance_loss_weight: float = 1.0,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    ood_noise_std: float = 0.1,
    lr_g: Optional[float] = None,
    feature_names: Optional[List[str]] = None,
) -> STGNN:
    """Train one SDE-Net ST-GNN (alternating drift / diffusion optimisation).

    The diffusion objective needs a pseudo-OOD batch: training inputs + Gaussian
    noise of std `ood_noise_std` (sin_elev/cos_elev excluded via feature_names).
    Per-epoch metrics are stored on `model.train_loss_history`.
    """
    if use_irradiance_loss:
        if getattr(model, "head_ghi", None) is None:
            raise ValueError(
                "use_irradiance_loss=True requires a model with an irradiance head "
                "(STGNN with use_irradiance_head=True); this model has no head_ghi."
            )
        if dataset.kt_target_all is None:
            raise ValueError(
                "use_irradiance_loss=True requires kt targets on the training "
                "dataset (build_datasets attaches them via kt_by_year)."
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

    kt_max = float(getattr(model, "KT_MAX", 1.2))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device)

    # Two optimisers (Algorithm 1): opt_f over the drift + encoder + GAT + heads,
    # opt_g over the diffusion net only.
    g_params = list(model.sde.diffusion_net.parameters())
    g_ids = {id(p) for p in g_params}
    f_params = [p for p in model.parameters() if id(p) not in g_ids]
    opt_f = torch.optim.AdamW(f_params, lr=lr, weight_decay=1e-4)
    opt_g = torch.optim.AdamW(g_params, lr=lr if lr_g is None else lr_g)

    loss_fn = make_loss_fn(loss_type, huber_delta)
    loss_label = "mse" if loss_type == "mse" else f"huber(delta={huber_delta})"

    def _kt_target(k):
        return torch.from_numpy(
            np.clip(dataset.kt_target_all[k.numpy()], 0.0, kt_max)
        ).to(device)

    history: List[dict] = []
    t_train = time.perf_counter()
    for ep in range(epochs):
        model.train()
        t_ep = time.perf_counter()
        losses, losses_pv, losses_irr = [], [], []
        g_in_list, g_ood_list = [], []
        for x, y, k in loader:
            x, y = x.to(device), y.to(device)

            # --- drift step: point loss on the in-distribution prediction ---
            pred_ghi, pred_pv = model(x, ei, ew, None, stochastic=True)
            loss_pv = loss_fn(pred_pv, y)
            loss = loss_pv
            if use_irradiance_loss:
                loss_irr = loss_fn(pred_ghi, _kt_target(k))
                loss = loss + irradiance_loss_weight * loss_irr
                losses_irr.append(float(loss_irr.item()))
            opt_f.zero_grad()
            loss.backward()
            opt_f.step()

            # --- diffusion step: g low in-distribution, high on Gaussian OOD ---
            x_ood = inject_input_noise(x, noise_idx_t, ood_noise_std, 1.0)
            with torch.no_grad():
                x0_in = model.encode(x, ei, ew)
                x0_ood = model.encode(x_ood, ei, ew)
            g_in = model.sde.diffusion(x0_in).mean()
            g_ood = model.sde.diffusion(x0_ood).mean()
            loss_g = g_in - g_ood  # minimise g_in, maximise g_ood
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            losses.append(float(loss.item()))
            losses_pv.append(float(loss_pv.item()))
            g_in_list.append(float(g_in.item()))
            g_ood_list.append(float(g_ood.item()))

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
        extra = (
            f"  g_in={g_in_m:.5f}  g_ood={g_ood_m:.5f}  g_ratio={rec['train/g_ratio']:.3f}"
        )
        if use_irradiance_loss:
            print(
                f"  [stgnn] epoch {ep + 1}/{epochs}  loss/total={rec['loss/total']:.5f}  "
                f"loss/pv={rec['loss/pv']:.5f}  loss/irradiance={rec['loss/irradiance']:.5f}  "
                f"(loss={loss_label}, weight={irradiance_loss_weight}){extra}  "
                f"[time] epoch: {time.perf_counter() - t_ep:.1f}s"
            )
        else:
            print(
                f"  [stgnn] epoch {ep + 1}/{epochs}  train_{loss_label}(norm)={np.mean(losses):.5f}"
                f"{extra}  [time] epoch: {time.perf_counter() - t_ep:.1f}s"
            )
        history.append(rec)
    model.train_loss_history = history
    print(f"  [stgnn] [time] train_model total: {time.perf_counter() - t_train:.1f}s")
    return model
