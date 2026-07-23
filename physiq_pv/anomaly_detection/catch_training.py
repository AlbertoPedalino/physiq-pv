"""Explicit bi-level optimisation for CATCH."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from physiq_pv.anomaly_detection.catch_model import (
    CATCHModel,
    CATCHModelOutput,
    frequency_reconstruction_loss,
)


@dataclass(frozen=True)
class CATCHTrainingConfig:
    """Optimisation settings separated from model architecture."""

    frequency_loss_weight: float = 0.005
    clustering_weight: float = 0.005
    regularization_weight: float = 0.0025
    batch_size: int = 32
    learning_rate: float = 1e-4
    mask_learning_rate: float = 1e-5
    epochs: int = 3
    patience: int = 3
    model_steps_per_mask: int | None = None
    gradient_clip: float | None = None
    lr_adjustment: str = "type1"
    minimum_oom_batch_size: int = 8
    seed: int = 42


class CATCHTrainer:
    """Paper-order mask-outer/model-inner CATCH trainer."""

    def __init__(
        self,
        model: CATCHModel,
        config: CATCHTrainingConfig,
        *,
        device: torch.device,
        verbose: bool = True,
    ) -> None:
        if config.batch_size < 1 or config.epochs < 1 or config.patience < 1:
            raise ValueError("batch_size, epochs, and patience must be >= 1")
        if (
            config.model_steps_per_mask is not None
            and config.model_steps_per_mask < 1
        ):
            raise ValueError("model_steps_per_mask must be >= 1")
        if config.gradient_clip is not None and config.gradient_clip <= 0:
            raise ValueError("gradient_clip must be > 0 when provided")
        if config.lr_adjustment not in {"type1", "constant"}:
            raise ValueError("lr_adjustment must be 'type1' or 'constant'")
        if config.minimum_oom_batch_size < 1:
            raise ValueError("minimum_oom_batch_size must be >= 1")
        self.model = model
        self.config = config
        self.device = device
        self.verbose = bool(verbose)
        self.history: list[dict[str, float]] = []
        self.effective_batch_size = int(config.batch_size)
        self.effective_model_steps_per_mask = 0

    @staticmethod
    def repository_model_steps_per_mask(loader_length: int) -> int:
        """Resolve ``N_I`` from the cadence used by the official repository."""

        if loader_length < 1:
            raise ValueError("loader_length must be >= 1")
        return min(max(loader_length // 10, 1), 100)

    def _learning_rate_for_epoch(self, base_rate: float, epoch: int) -> float:
        """Mirror the repository's default ``type1`` epoch schedule."""

        if self.config.lr_adjustment == "constant":
            return float(base_rate)
        return float(base_rate * (0.5 ** max(epoch - 1, 0)))

    @staticmethod
    def _set_learning_rate(
        optimizer: torch.optim.Optimizer, learning_rate: float
    ) -> None:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    @staticmethod
    def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    def _loss_terms(
        self, batch: torch.Tensor, output: CATCHModelOutput
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized, _, _ = self.model.revin.normalize(batch)
        time_loss = F.mse_loss(output.reconstruction, batch)
        frequency_loss = frequency_reconstruction_loss(
            output.frequency_reconstruction, normalized
        )
        total = (
            time_loss
            + self.config.frequency_loss_weight * frequency_loss
            + self.config.clustering_weight * output.clustering_loss
            + self.config.regularization_weight * output.regularization_loss
        )
        return total, {
            "time_loss": time_loss,
            "frequency_loss": frequency_loss,
            "clustering_loss": output.clustering_loss,
            "regularization_loss": output.regularization_loss,
        }

    def _set_trainable_group(self, *, mask_only: bool) -> list[nn.Parameter]:
        selected: list[nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            is_mask = name.startswith("mask_generator.")
            parameter.requires_grad_(is_mask if mask_only else not is_mask)
            if parameter.requires_grad:
                selected.append(parameter)
        return selected

    def _mask_step(
        self, batch: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        parameters = self._set_trainable_group(mask_only=True)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = self._loss_terms(batch, self.model(batch))
        loss.backward()
        if self.config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
        optimizer.step()

    def _model_step(
        self, batch: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> tuple[float, dict[str, float]]:
        parameters = self._set_trainable_group(mask_only=False)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = self._loss_terms(batch, self.model(batch))
        loss.backward()
        if self.config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
        optimizer.step()
        return float(loss.detach()), {
            name: float(value.detach()) for name, value in terms.items()
        }

    def _validation_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch, _, _ in loader:
                batch = batch.to(self.device)
                losses.append(float(F.mse_loss(self.model(batch).reconstruction, batch)))
        return float(np.mean(losses))

    def _fit_once(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        *,
        batch_size: int,
    ) -> list[dict[str, float]]:
        generator = torch.Generator().manual_seed(self.config.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        model_steps_per_mask = (
            self.config.model_steps_per_mask
            if self.config.model_steps_per_mask is not None
            else self.repository_model_steps_per_mask(len(train_loader))
        )
        self.effective_model_steps_per_mask = int(model_steps_per_mask)
        main_parameters = self._set_trainable_group(mask_only=False)
        main_optimizer = torch.optim.Adam(
            main_parameters, lr=self.config.learning_rate
        )
        mask_parameters = self._set_trainable_group(mask_only=True)
        mask_optimizer = torch.optim.Adam(
            mask_parameters, lr=self.config.mask_learning_rate
        )

        best_loss = float("inf")
        best_state = self._cpu_state_dict(self.model)
        stale_epochs = 0
        self.history = []
        for epoch in range(self.config.epochs):
            main_learning_rate = self._learning_rate_for_epoch(
                self.config.learning_rate, epoch
            )
            mask_learning_rate = self._learning_rate_for_epoch(
                self.config.mask_learning_rate, epoch
            )
            self._set_learning_rate(main_optimizer, main_learning_rate)
            self._set_learning_rate(mask_optimizer, mask_learning_rate)
            self.model.train()
            epoch_terms: dict[str, list[float]] = {
                "loss": [],
                "time_loss": [],
                "frequency_loss": [],
                "clustering_loss": [],
                "regularization_loss": [],
            }
            iterator = iter(train_loader)
            exhausted = False
            while not exhausted:
                try:
                    outer_batch, _, _ = next(iterator)
                except StopIteration:
                    break
                outer_batch = outer_batch.to(self.device)
                self._mask_step(outer_batch, mask_optimizer)

                for inner_index in range(model_steps_per_mask):
                    if inner_index == 0:
                        inner_batch = outer_batch
                    else:
                        try:
                            inner_batch, _, _ = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                        inner_batch = inner_batch.to(self.device)
                    loss, terms = self._model_step(inner_batch, main_optimizer)
                    epoch_terms["loss"].append(loss)
                    for name, value in terms.items():
                        epoch_terms[name].append(value)

            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            valid_loss = self._validation_loss(validation_loader)
            record = {
                "epoch": float(epoch + 1),
                "batch_size": float(batch_size),
                "model_steps_per_mask": float(model_steps_per_mask),
                "learning_rate": main_learning_rate,
                "mask_learning_rate": mask_learning_rate,
                **{
                    name: float(np.mean(values))
                    for name, values in epoch_terms.items()
                },
                "valid_loss": valid_loss,
            }
            self.history.append(record)
            if self.verbose:
                print(
                    f"epoch={epoch + 1} loss={record['loss']:.6f} "
                    f"valid={valid_loss:.6f}"
                )
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = self._cpu_state_dict(self.model)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        self.model.load_state_dict(best_state)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        return list(self.history)

    def fit(
        self, train_dataset: Dataset, validation_dataset: Dataset
    ) -> list[dict[str, float]]:
        """Run Algorithm 1, retrying CUDA OOMs with the paper's batch policy."""

        initial_state = self._cpu_state_dict(self.model)
        batch_size = int(self.config.batch_size)
        minimum_batch_size = min(
            batch_size, int(self.config.minimum_oom_batch_size)
        )
        while True:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            self.model.load_state_dict(initial_state)
            self.effective_batch_size = batch_size
            try:
                return self._fit_once(
                    train_dataset,
                    validation_dataset,
                    batch_size=batch_size,
                )
            except RuntimeError as error:
                is_cuda_oom = (
                    self.device.type == "cuda"
                    and "out of memory" in str(error).lower()
                )
                if not is_cuda_oom or batch_size <= minimum_batch_size:
                    raise
                batch_size = max(minimum_batch_size, batch_size // 2)
                error.__traceback__ = None
                if self.verbose:
                    print(
                        "CUDA OOM: restarting CATCH training with "
                        f"batch_size={batch_size}"
                    )
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
