import numpy as np
import xarray as xr
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    model_name: str
    mae_ghi: float = 0.0
    mae_pv: float = 0.0
    rmse_ghi: float = 0.0
    rmse_pv: float = 0.0
    detection_rate: float = 0.0
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.model_name:<28} "
            f"MAE_PV={self.mae_pv:.4f}  RMSE_PV={self.rmse_pv:.4f}  "
            f"MAE_GHI={self.mae_ghi:.4f}  DetRate={self.detection_rate:.3f}"
        )


def _metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    mask = ~(np.isnan(pred) | np.isnan(true))
    if mask.sum() == 0:
        return float("nan"), float("nan")
    diff = pred[mask] - true[mask]
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    return mae, rmse


class AblationBenchmark:
    """Collects and reports ablation results."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def report(self) -> str:
        if not self.results:
            return "No results recorded."
        header = f"{'Model':<28} {'MAE_PV':>10} {'RMSE_PV':>10} {'MAE_GHI':>10} {'DetRate':>9}"
        sep = "-" * len(header)
        rows = [header, sep]
        for r in sorted(self.results, key=lambda x: x.mae_pv):
            rows.append(
                f"{r.model_name:<28} {r.mae_pv:>10.4f} {r.rmse_pv:>10.4f} "
                f"{r.mae_ghi:>10.4f} {r.detection_rate:>9.3f}"
            )
        return "\n".join(rows)


class NaiveBaseline:
    """
    Irradiance baseline: scale POA irradiance to each plant's observed p99 PV.
    Used to verify that ST-GNN beats trivial forecasts.
    """

    def predict(self, ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        pred_ghi = np.clip(ds["solar_irradiance_poa"].values / 1000.0, 0.0, None)
        true_pv = ds["ENERGIA"].values.astype(float)
        n_plants = true_pv.shape[0]
        pred_pv = np.zeros_like(true_pv, dtype=float)

        for p in range(n_plants):
            day = pred_ghi[p] > 0.05
            valid = day & np.isfinite(true_pv[p]) & (true_pv[p] > 0)
            if valid.sum() > 10:
                pv_p99 = np.percentile(true_pv[p][valid], 99)
                ghi_p99 = np.percentile(pred_ghi[p][valid], 99)
                scale = pv_p99 / ghi_p99 if ghi_p99 > 0 else 0.0
            else:
                scale = 0.0
            pred_pv[p] = pred_ghi[p] * scale

        return pred_ghi, pred_pv

    def evaluate(self, ds: xr.Dataset) -> BenchmarkResult:
        pred_ghi, pred_pv = self.predict(ds)
        true_pv = ds["ENERGIA"].values
        true_ghi = ds["solar_irradiance_poa"].values / 1000.0
        mae_pv, rmse_pv = _metrics(pred_pv.ravel(), true_pv.ravel())
        mae_ghi, rmse_ghi = _metrics(pred_ghi.ravel(), true_ghi.ravel())
        return BenchmarkResult(
            model_name="NaiveBaseline",
            mae_ghi=mae_ghi, rmse_ghi=rmse_ghi,
            mae_pv=mae_pv, rmse_pv=rmse_pv,
        )
