"""Explicit bi-level optimisation for CATCH."""

from __future__ import annotations

from copy import deepcopy
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
    batch_size: int = 128
    learning_rate: float = 1e-4
    mask_learning_rate: float = 1e-5
    epochs: int = 3
    patience: int = 3
    model_steps_per_mask: int = 10
    gradient_clip: float = 1.0
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
        if config.model_steps_per_mask < 1:
            raise ValueError("model_steps_per_mask must be >= 1")
        self.model = model
        self.config = config
        self.device = device
        self.verbose = bool(verbose)
        self.history: list[dict[str, float]] = []

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
        nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
        optimizer.step()

    def _model_step(
        self, batch: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> tuple[float, dict[str, float]]:
        parameters = self._set_trainable_group(mask_only=False)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = self._loss_terms(batch, self.model(batch))
        loss.backward()
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

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset) -> list[dict[str, float]]:
        """Update one mask outer step before each group of model inner steps."""

        generator = torch.Generator().manual_seed(self.config.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        main_parameters = self._set_trainable_group(mask_only=False)
        main_optimizer = torch.optim.Adam(
            main_parameters, lr=self.config.learning_rate
        )
        mask_parameters = self._set_trainable_group(mask_only=True)
        mask_optimizer = torch.optim.Adam(
            mask_parameters, lr=self.config.mask_learning_rate
        )

        best_loss = float("inf")
        best_state = deepcopy(self.model.state_dict())
        stale_epochs = 0
        self.history = []
        for epoch in range(self.config.epochs):
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

                for inner_index in range(self.config.model_steps_per_mask):
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
                best_state = deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        self.model.load_state_dict(best_state)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        return list(self.history)
