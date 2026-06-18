"""Training loop for the PVGIS ST-GNN run (Huber + MC-penalty + train noise)."""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from physiq_pv.data.pvgis_dataset import PVGISWindowDataset
from physiq_pv.model.st_gnn import STGNN
from physiq_pv.training.losses import (
    make_loss_fn,
    sde_proxy_penalty,
)
from physiq_pv.training.noise import (
    build_noise_feature_indices,
    inject_input_noise,
    inject_input_noise_anomaly,
)


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
    train_mc_uncertainty_penalty: bool = False,
    train_mc_samples: int = 1,
    uncertainty_penalty_mode: str = "sde_proxy",
    sde_proxy_in_weight: float = 0.001,
    sde_proxy_out_weight: float = 0.1,
    sde_proxy_std_min_ood: float = 0.05,
    train_noise_std: float = 0.0,
    train_noise_prob: float = 0.0,
    train_noise_mode: str = "random",
    anomaly_noise_std: float = 0.0,
    anomaly_noise_prob: float = 0.0,
    feature_names: Optional[List[str]] = None,
) -> STGNN:
    """Train pred_pv against the normalised PVGIS pv target. Deterministic, no QS.

    Default (all new flags off) is the historical behaviour: a single forward
    pass per batch, point loss L(pred_pv, y) with L = MSE or Huber (loss_type),
    plus the optional kt-aux term. The irradiance head receives gradient only
    with use_irradiance_loss=True.

    train_mc_uncertainty_penalty=True turns on a train-time MC uncertainty
    penalty: each batch runs `train_mc_samples` STOCHASTIC forward passes
    (model.train() keeps dropout active), giving a per-target MC mean and std.
    Needs train_mc_samples >= 2 and dropout > 0. The penalty is applied to the PV
    target ONLY; the kt-aux term is left as-is (computed on the MC-mean kt head,
    same loss module). The penalty (uncertainty_penalty_mode='sde_proxy'):

      * "sde_proxy": SDE-Net-style — MINIMISE uncertainty on in-distribution
        cells, KEEP it above a floor on OOD/anomalous cells (y_pred_std is the
        diffusion proxy; there is no explicit g(x)). Uses the per-(sample,node)
        anomaly mask (normal = in-dist, rare_or_extreme = OOD):
            in_loss  = mean(std[normal]^2)
            out_loss = mean(relu(std_min_ood - std[anomaly])^2)
            loss     = L(mean,y) + sde_proxy_in_weight*in_loss
                                 + sde_proxy_out_weight*out_loss
        Only supported with train_noise_mode="anomaly" (needs the OOD mask) and
        requires at least one anomalous training cell. std_min_ood is in the
        NORMALISED target scale.

    Input noise injection (train only; targets and eval/inference untouched;
    requires feature_names to map channels):
      * train_noise_mode="random" (default): N(0, train_noise_std) on the
        continuous channels (sin_elev/cos_elev excluded) of a
        Bernoulli(train_noise_prob) fraction of TRAIN samples;
      * train_noise_mode="anomaly": ANOMALY-AWARE noise. Cells flagged in
        dataset.anomaly_mask_all (rare_or_extreme at the target time) use
        (anomaly_noise_std, anomaly_noise_prob); the rest use the random pair
        (train_noise_std, train_noise_prob). Requires dataset.anomaly_mask_all
        with at least one anomalous cell, else a ValueError is raised. NOTE: this
        couples TRAINING to anomaly labels — it is no longer an eval-only-labels
        configuration.

    Apart from the anomaly-noise mask, stratification labels never enter this
    training path. Per-epoch loss components are stored on
    `model.train_loss_history`.
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
                "irradiance_loss_weight must be finite and >= 0, got "
                f"{irradiance_loss_weight}."
            )

    penalty = bool(train_mc_uncertainty_penalty)
    if penalty:
        if train_mc_samples < 2:
            raise ValueError(
                "train_mc_uncertainty_penalty=True needs train_mc_samples >= 2 "
                f"(a per-target std needs >= 2 MC passes); got {train_mc_samples}."
            )
    if uncertainty_penalty_mode != "sde_proxy":
        raise ValueError(
            "uncertainty_penalty_mode must be 'sde_proxy', "
            f"got {uncertainty_penalty_mode!r}."
        )
    sde_proxy_mode = uncertainty_penalty_mode == "sde_proxy"
    # The mode only matters when the MC penalty is enabled (sde_proxy is now the
    # only mode, so it is the default; a penalty-off run is plain training).
    sde_proxy_active = bool(penalty and sde_proxy_mode)
    if sde_proxy_active:
        for nm, w in (
            ("sde_proxy_in_weight", sde_proxy_in_weight),
            ("sde_proxy_out_weight", sde_proxy_out_weight),
        ):
            if not np.isfinite(w) or w < 0.0:
                raise ValueError(f"{nm} must be finite and >= 0, got {w}.")
        if not np.isfinite(sde_proxy_std_min_ood) or sde_proxy_std_min_ood < 0.0:
            raise ValueError(
                "sde_proxy_std_min_ood must be finite and >= 0, got "
                f"{sde_proxy_std_min_ood}."
            )

    if train_noise_mode not in ("random", "anomaly"):
        raise ValueError(
            f"train_noise_mode must be 'random' or 'anomaly', got {train_noise_mode!r}."
        )
    random_noise_active = train_noise_std > 0.0 and train_noise_prob > 0.0
    anomaly_mode = train_noise_mode == "anomaly"
    anomaly_noise_active = anomaly_mode and (
        anomaly_noise_std > 0.0 and anomaly_noise_prob > 0.0
    )
    noise_active = random_noise_active or anomaly_noise_active
    anomaly_mask_all = None
    noise_idx_t = None
    if anomaly_mode:
        # The anomaly mask must be CONSUMED by something: anomaly-aware noise
        # and/or the sde_proxy penalty. (sde_proxy alone, with noise off, is OK.)
        if not anomaly_noise_active and not sde_proxy_active:
            raise ValueError(
                "train_noise_mode='anomaly' needs anomaly_noise_std > 0 and "
                f"anomaly_noise_prob > 0 (got std={anomaly_noise_std}, "
                f"prob={anomaly_noise_prob}), unless uncertainty_penalty_mode="
                "'sde_proxy' consumes the mask."
            )
        if anomaly_noise_active and not (0.0 <= anomaly_noise_prob <= 1.0):
            raise ValueError(
                f"anomaly_noise_prob must be in [0, 1], got {anomaly_noise_prob}."
            )
        anomaly_mask_all = getattr(dataset, "anomaly_mask_all", None)
        if anomaly_mask_all is None:
            raise ValueError(
                "train_noise_mode='anomaly' requires anomaly labels on the TRAINING "
                "dataset: call dataset.attach_anomaly_mask(scores) with anomaly "
                "scores covering the training years before train_model."
            )
        if not bool(np.any(anomaly_mask_all)):
            raise ValueError(
                "train_noise_mode='anomaly' but the training dataset has NO "
                "anomalous (rare_or_extreme) cells — the provided anomaly scores "
                "do not cover any training (location, target_time). Provide "
                "train-year anomaly scores or use train_noise_mode='random'."
            )
    if noise_active:
        if not (0.0 <= train_noise_prob <= 1.0):
            raise ValueError(
                f"train_noise_prob must be in [0, 1], got {train_noise_prob}."
            )
        if feature_names is None:
            raise ValueError(
                "input noise injection requires feature_names to map the noise "
                "channels (sin_elev/cos_elev are excluded)."
            )
        noise_idx = build_noise_feature_indices(feature_names)
        if not noise_idx:
            raise ValueError(
                "no continuous feature channels available for noise injection "
                f"(feature_names={feature_names} after excluding "
                f"{NOISE_EXCLUDED_FEATURES})."
            )
        noise_idx_t = torch.tensor(noise_idx, dtype=torch.long, device=device)

    if sde_proxy_active and not anomaly_mode:
        # The OOD/in-distribution split comes from the anomaly mask, which is
        # only attached/validated under train_noise_mode='anomaly'.
        raise ValueError(
            "uncertainty_penalty_mode='sde_proxy' is only supported with "
            "train_noise_mode='anomaly' for now (it needs the OOD/anomaly mask). "
            "Use train_noise_mode='anomaly' with train-year anomaly scores."
        )

    kt_max = float(getattr(model, "KT_MAX", 1.2))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        mean_stds = []
        sde_in, sde_out, std_normal_list, std_anom_list = [], [], [], []
        for x, y, k in loader:
            x, y = x.to(device), y.to(device)
            mask_batch = None
            if anomaly_mode:
                mask_batch = torch.from_numpy(
                    anomaly_mask_all[k.numpy()]
                ).to(device=device, dtype=torch.bool)  # (B, N)
            if noise_active:
                if anomaly_mode:
                    x = inject_input_noise_anomaly(
                        x, noise_idx_t, mask_batch,
                        train_noise_std, train_noise_prob,
                        anomaly_noise_std, anomaly_noise_prob,
                    )
                else:
                    x = inject_input_noise(
                        x, noise_idx_t, train_noise_std, train_noise_prob
                    )
            if penalty:
                pv_samples, ghi_samples = [], []
                for _ in range(train_mc_samples):
                    g_s, p_s = model(x, ei, ew, None)
                    pv_samples.append(p_s)
                    if use_irradiance_loss:
                        ghi_samples.append(g_s)
                pv_stack = torch.stack(pv_samples, dim=0)        # (S, B, N)
                y_pred_mean = pv_stack.mean(dim=0)               # (B, N)
                y_pred_std = pv_stack.std(dim=0, unbiased=False)  # (B, N)
                loss_pv = loss_fn(y_pred_mean, y)
                if sde_proxy_mode:
                    # SDE-Net proxy: minimise std in-distribution, keep std above
                    # a floor on OOD/anomalous cells. Mask: True = OOD/anomaly.
                    in_loss, out_loss = sde_proxy_penalty(
                        y_pred_std, mask_batch, sde_proxy_std_min_ood
                    )
                    loss = (
                        loss_pv
                        + sde_proxy_in_weight * in_loss
                        + sde_proxy_out_weight * out_loss
                    )
                    normal_m = ~mask_batch
                    sde_in.append(float(in_loss.item()))
                    sde_out.append(float(out_loss.item()))
                    std_normal_list.append(
                        float(y_pred_std[normal_m].mean().item())
                        if bool(normal_m.any()) else float("nan")
                    )
                    std_anom_list.append(
                        float(y_pred_std[mask_batch].mean().item())
                        if bool(mask_batch.any()) else float("nan")
                    )
                mean_stds.append(float(y_pred_std.mean().item()))
                if use_irradiance_loss:
                    pred_ghi_mean = torch.stack(ghi_samples, dim=0).mean(dim=0)
                    loss_irr = loss_fn(pred_ghi_mean, _kt_target(k))
                    loss = loss + irradiance_loss_weight * loss_irr
                    losses_irr.append(float(loss_irr.item()))
            else:
                pred_ghi, pred_pv = model(x, ei, ew, None)  # ghi_cs=None -> pred_kt
                loss_pv = loss_fn(pred_pv, y)
                if use_irradiance_loss:
                    loss_irr = loss_fn(pred_ghi, _kt_target(k))
                    loss = loss_pv + irradiance_loss_weight * loss_irr
                    losses_irr.append(float(loss_irr.item()))
                else:
                    loss = loss_pv
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            losses_pv.append(float(loss_pv.item()))
        rec = {
            "loss/total": float(np.mean(losses)),
            "loss/pv": float(np.mean(losses_pv)),
        }
        if use_irradiance_loss:
            rec["loss/irradiance"] = float(np.mean(losses_irr))
        extra = ""
        if penalty:
            rec["train/mean_pred_std"] = float(np.mean(mean_stds))
            if sde_proxy_mode:
                rec["train/uncertainty_in_loss"] = float(np.mean(sde_in))
                rec["train/uncertainty_out_loss"] = float(np.mean(sde_out))
                msn = float(np.nanmean(std_normal_list)) if std_normal_list else float("nan")
                msa = float(np.nanmean(std_anom_list)) if std_anom_list else float("nan")
                rec["train/mean_std_normal"] = msn
                rec["train/mean_std_anomaly"] = msa
                rec["train/std_ratio_anomaly_vs_normal"] = (
                    float(msa / msn) if (np.isfinite(msn) and msn > 0.0) else float("nan")
                )
                extra = (
                    f"  in_loss={rec['train/uncertainty_in_loss']:.6f}"
                    f"  out_loss={rec['train/uncertainty_out_loss']:.6f}"
                    f"  std_norm={msn:.5f}  std_anom={msa:.5f}"
                    f"  std_ratio={rec['train/std_ratio_anomaly_vs_normal']:.3f}"
                    f"  (mc={train_mc_samples}, mode=sde_proxy)"
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
