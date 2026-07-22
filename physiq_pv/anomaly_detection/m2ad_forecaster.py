"""Windowing and stacked-LSTM forecasting component for M2AD."""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class M2ADWindowDataset(Dataset):
    """Lazy windows over disjoint segments; windows never cross boundaries."""

    def __init__(self, segments: Sequence[np.ndarray], window_size: int, horizon: int = 1):
        self.segments = [np.asarray(segment, dtype=np.float32) for segment in segments]
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        if self.window_size < 1 or self.horizon < 1:
            raise ValueError("window_size and horizon must both be >= 1")

        segment_ids: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        for segment_id, array in enumerate(self.segments):
            count = array.shape[0] - self.window_size - self.horizon + 1
            if count <= 0:
                continue
            segment_ids.append(np.full(count, segment_id, dtype=np.int32))
            starts.append(np.arange(count, dtype=np.int32))
        if not starts:
            raise ValueError(
                "no windows available; segments are shorter than window_size + horizon"
            )
        self.segment_ids = np.concatenate(segment_ids)
        self.starts = np.concatenate(starts)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int):
        segment_id = int(self.segment_ids[index])
        start = int(self.starts[index])
        target_index = start + self.window_size + self.horizon - 1
        segment = self.segments[segment_id]
        window = torch.from_numpy(
            np.ascontiguousarray(segment[start : start + self.window_size])
        )
        target = torch.from_numpy(np.ascontiguousarray(segment[target_index]))
        return window, target, segment_id, target_index


class M2ADLSTM(nn.Module):
    """Stacked forecasting LSTM used by the public M2AD implementation."""

    def __init__(
        self,
        n_sensors: int,
        hidden_size: int = 80,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_sensors < 1 or hidden_size < 1 or n_layers < 1:
            raise ValueError("n_sensors, hidden_size, and n_layers must be >= 1")
        self.lstm1 = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, n_sensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        _, (hidden, _) = self.lstm2(x)
        return self.output(hidden[-1])


class M2ADForecaster:
    """Own LSTM construction, optimization, early stopping, and prediction."""

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_size: int = 80,
        n_layers: int = 2,
        dropout: float = 0.2,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        validation_split: float = 0.2,
        patience: int = 5,
        min_delta: float = 0.0,
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.n_sensors = int(n_sensors)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.validation_split = float(validation_split)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.device = str(device)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.patience < 0:
            raise ValueError("patience must be >= 0")
        self.model: M2ADLSTM | None = None
        self.history: list[dict[str, float]] = []

    def _new_model(self) -> M2ADLSTM:
        return M2ADLSTM(
            self.n_sensors,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

    def fit(self, dataset: M2ADWindowDataset) -> "M2ADForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model = self._new_model()

        n_total = len(dataset)
        n_valid = int(n_total * self.validation_split)
        if self.validation_split > 0 and n_total > 1:
            n_valid = max(1, min(n_valid, n_total - 1))
        else:
            n_valid = 0
        split = n_total - n_valid
        train_set = Subset(dataset, range(0, split))
        valid_set = Subset(dataset, range(split, n_total)) if n_valid else None
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=0,
        )
        valid_loader = (
            DataLoader(valid_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
            if valid_set is not None
            else None
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        best_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        stale_epochs = 0
        self.history = []

        for epoch in range(self.epochs):
            self.model.train()
            train_losses: list[float] = []
            for windows, targets, _, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(
                    self.model(windows.to(self.device)), targets.to(self.device)
                )
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            self.model.eval()
            valid_losses: list[float] = []
            if valid_loader is not None:
                with torch.no_grad():
                    for windows, targets, _, _ in valid_loader:
                        prediction = self.model(windows.to(self.device))
                        valid_losses.append(
                            float(criterion(prediction, targets.to(self.device)).cpu())
                        )
            train_loss = float(np.mean(train_losses))
            valid_loss = float(np.mean(valid_losses)) if valid_losses else train_loss
            self.history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss}
            )
            if self.verbose:
                print(
                    f"    epoch {epoch + 1:02d}/{self.epochs}: "
                    f"train={train_loss:.6f} valid={valid_loss:.6f}"
                )
            if valid_loss < best_loss - self.min_delta:
                best_loss = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if self.patience > 0 and stale_epochs >= self.patience:
                    if self.verbose:
                        print(f"    early stopping after epoch {epoch + 1}")
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict(self, dataset: M2ADWindowDataset) -> tuple[np.ndarray, ...]:
        if self.model is None:
            raise RuntimeError("forecaster is not fitted")
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        observed: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        segment_ids: list[np.ndarray] = []
        target_indices: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for windows, targets, segment, target_index in loader:
                predicted.append(self.model(windows.to(self.device)).cpu().numpy())
                observed.append(targets.numpy())
                segment_ids.append(segment.numpy())
                target_indices.append(target_index.numpy())
        return (
            np.concatenate(observed),
            np.concatenate(predicted),
            np.concatenate(segment_ids).astype(np.int32),
            np.concatenate(target_indices).astype(np.int64),
        )
