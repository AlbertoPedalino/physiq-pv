import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


class QSClusterer:
    """
    Soft-DTW TimeSeriesKMeans on per-plant QS(t) trajectories.
    Groups plants by degradation pattern for spatial diagnosis.

    Reference: tslearn-team/tslearn
    """

    def __init__(self, n_clusters: int = 4, metric: str = "softdtw", random_state: int = 42):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.model: TimeSeriesKMeans | None = None
        self.labels_: np.ndarray | None = None
        self._scaler = TimeSeriesScalerMeanVariance()

    def fit(self, qs_matrix: np.ndarray) -> "QSClusterer":
        """
        qs_matrix: (N_plants, T) — daily or hourly QS values.
        Sets self.labels_ (N_plants,).
        """
        qs_3d = self._scaler.fit_transform(qs_matrix[:, :, np.newaxis])  # (N, T, 1)
        self.model = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric=self.metric,
            max_iter=50,
            random_state=self.random_state,
            verbose=False,
        )
        self.labels_ = self.model.fit_predict(qs_3d)
        return self

    def predict(self, qs_matrix: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Call fit() first."
        qs_3d = self._scaler.transform(qs_matrix[:, :, np.newaxis])
        return self.model.predict(qs_3d)

    def cluster_summary(self) -> dict[int, dict]:
        assert self.labels_ is not None, "Call fit() first."
        return {
            c: {
                "plants": list(np.where(self.labels_ == c)[0].astype(int)),
                "size": int((self.labels_ == c).sum()),
            }
            for c in range(self.n_clusters)
        }
