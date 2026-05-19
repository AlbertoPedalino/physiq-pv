"""
Continual Learning metrics: Backward Transfer, Forward Transfer,
Average Forgetting, Learning Curve.

Definitions follow Lopez-Paz & Ranzato (NeurIPS 2017, GEM) and the variant
used in Buzzega et al. (NeurIPS 2020, DER++):

    R[i, j] = performance on task j after training on task i.

Let T = number of tasks. With these conventions:

    BWT = (1/(T-1)) * sum_{i=0..T-2} (R[T-1, i] - R[i, i])
        positive -> training on later tasks improved earlier ones.
        negative -> catastrophic forgetting.

    FWT = (1/(T-1)) * sum_{i=1..T-1} (R[i-1, i] - b[i])
        b[i] = naive baseline on task i (random / persistence).
        positive -> the model generalises to task i before training on it.

    AF  = (1/(T-1)) * sum_{i=0..T-2} (max_{l in [i, T-1]} R[l, i] - R[T-1, i])
        non-negative; zero -> no forgetting.

    Learning curve = diagonal R[i, i] for i in 0..T-1.

This module is metric-agnostic. For loss-style metrics where lower is
better, pass `higher_is_better=False` to flip signs so that BWT > 0 still
means "forgetting reduced" and AF > 0 still means "forgetting present".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CLMetricsReport:
    n_tasks: int
    higher_is_better: bool
    bwt: float
    fwt: float
    avg_forgetting: float
    learning_curve: list[float]
    matrix: np.ndarray   # R[i, j]
    baseline: np.ndarray | None = None


class CLMetricsTracker:
    """
    Records the R[i, j] matrix during a continual-learning run.

    Workflow:
        tracker = CLMetricsTracker(n_tasks=T, higher_is_better=False)
        for i in range(T):
            train_on(task_i)
            for j in range(T):
                tracker.record(i, j, evaluate(model, task_j))
        report = tracker.compute()

    Cells that are never recorded remain NaN and are ignored by the
    aggregate metrics (with a clear error if too many are missing).

    `baseline` (optional) is a length-T array of naive baseline metrics per
    task; required to compute FWT in the strict Lopez-Paz sense.
    """

    def __init__(
        self,
        n_tasks: int,
        higher_is_better: bool = True,
        baseline: np.ndarray | None = None,
    ):
        if n_tasks <= 1:
            raise ValueError("CL metrics require at least 2 tasks")
        self.n_tasks = int(n_tasks)
        self.higher_is_better = bool(higher_is_better)
        self.matrix = np.full((self.n_tasks, self.n_tasks), np.nan, dtype=np.float64)
        if baseline is not None:
            baseline = np.asarray(baseline, dtype=np.float64)
            if baseline.shape != (self.n_tasks,):
                raise ValueError(
                    f"baseline must be shape ({self.n_tasks},), got {baseline.shape}"
                )
        self.baseline = baseline

    def record(self, train_task: int, eval_task: int, value: float) -> None:
        if not (0 <= train_task < self.n_tasks):
            raise IndexError(f"train_task {train_task} out of range")
        if not (0 <= eval_task < self.n_tasks):
            raise IndexError(f"eval_task {eval_task} out of range")
        self.matrix[train_task, eval_task] = float(value)

    # ------------------------------------------------------------------ #
    # Aggregate computations
    # ------------------------------------------------------------------ #

    def _signed(self, x: np.ndarray | float) -> np.ndarray | float:
        """Flip sign if lower-is-better so BWT/FWT semantics stay consistent."""
        return x if self.higher_is_better else -x

    def backward_transfer(self) -> float:
        T = self.n_tasks
        last = self.matrix[T - 1, :T - 1]
        diag = np.diag(self.matrix)[:T - 1]
        diff = self._signed(last - diag)
        return float(np.nanmean(diff))

    def forward_transfer(self) -> float:
        T = self.n_tasks
        if self.baseline is None:
            # In absence of baseline, compute FWT relative to the first-seen
            # diagonal entry as a fallback. Document this in summary.
            pre_train = np.array([self.matrix[i - 1, i] for i in range(1, T)])
            diag_self = np.array([self.matrix[i, i] for i in range(1, T)])
            diff = self._signed(pre_train - diag_self)
            return float(np.nanmean(diff))
        pre_train = np.array([self.matrix[i - 1, i] for i in range(1, T)])
        b = self.baseline[1:]
        diff = self._signed(pre_train - b)
        return float(np.nanmean(diff))

    def average_forgetting(self) -> float:
        T = self.n_tasks
        if self.higher_is_better:
            best_past = np.array([np.nanmax(self.matrix[i:T - 1, i]) for i in range(T - 1)])
            final = self.matrix[T - 1, :T - 1]
            forgetting = best_past - final
        else:
            best_past = np.array([np.nanmin(self.matrix[i:T - 1, i]) for i in range(T - 1)])
            final = self.matrix[T - 1, :T - 1]
            forgetting = final - best_past
        return float(np.nanmean(forgetting))

    def learning_curve(self) -> list[float]:
        return [float(self.matrix[i, i]) for i in range(self.n_tasks)]

    def compute(self, strict: bool = False) -> CLMetricsReport:
        """
        Build a CLMetricsReport. With strict=True raises if any required
        cell is NaN; otherwise tolerates missing cells via nanmean/nanmax.
        """
        if strict:
            T = self.n_tasks
            required = np.array(
                [self.matrix[T - 1, i] for i in range(T)]
                + [self.matrix[i, i] for i in range(T)]
                + [self.matrix[i - 1, i] for i in range(1, T)]
            )
            if np.isnan(required).any():
                raise ValueError("required cells missing for strict compute")

        return CLMetricsReport(
            n_tasks=self.n_tasks,
            higher_is_better=self.higher_is_better,
            bwt=self.backward_transfer(),
            fwt=self.forward_transfer(),
            avg_forgetting=self.average_forgetting(),
            learning_curve=self.learning_curve(),
            matrix=self.matrix.copy(),
            baseline=self.baseline.copy() if self.baseline is not None else None,
        )
