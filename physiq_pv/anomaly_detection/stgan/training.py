"""Equivalent STGAN alternating updates, with optional diagnostic scopes."""
from contextlib import nullcontext
import torch
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
from .precision import autocast_context
from .model import masked_cell_mean


TERM_NAMES = ("generator_reconstruction_loss", "generator_reconstruction_weighted_loss",
              "generator_adversarial_loss", "discriminator_real_loss", "discriminator_fake_loss",
              "discriminator_real_mean", "discriminator_fake_mean")


class DeviceLossTotals:
    """Match the old float64 weighted accumulation without per-step host reads."""
    def __init__(self, device):
        self.totals = torch.zeros(2, dtype=torch.float64, device=device)
        self.samples = 0
        self._logged_totals = [0.0, 0.0]
        self._logged_samples = 0
        self.term_totals = None

    def update(self, generator_loss, discriminator_loss, count, terms=None):
        self.totals.add_(torch.stack((generator_loss.detach(), discriminator_loss.detach())).to(torch.float64) * count)
        self.samples += count
        if terms is not None:
            values = torch.stack([terms[name] for name in TERM_NAMES]).to(torch.float64) * count
            self.term_totals = values if self.term_totals is None else self.term_totals + values

    def means(self):
        return (self.totals / self.samples).cpu().tolist()

    def term_means(self):
        """Sample-weighted epoch means of the separate loss terms and of D's outputs."""
        if self.term_totals is None:
            return {}
        return dict(zip(TERM_NAMES, (self.term_totals / self.samples).cpu().tolist()))

    def means_since_last_log(self):
        """One host transfer per log; no extra per-batch device accumulator."""
        count = self.samples - self._logged_samples
        if count <= 0:
            raise ValueError("No new samples since the previous log.")
        current = self.totals.cpu().tolist()
        means = [(value - previous) / count
                 for value, previous in zip(current, self._logged_totals)]
        self._logged_totals, self._logged_samples = current, self.samples
        return means


def _mean_loss(losses):
    """Mean over the optimizer steps of one batch; a single step is returned unchanged."""
    return losses[0] if len(losses) == 1 else torch.stack(losses).mean()


def _terms(generator_terms, discriminator_terms):
    """TERM_NAMES values: each network's terms averaged over its own optimizer steps of the batch."""
    values = torch.cat((torch.stack(generator_terms).mean(dim=0), torch.stack(discriminator_terms).mean(dim=0)))
    return dict(zip(TERM_NAMES, values))


def gan_train_step(model, batch, generator_optimizer, discriminator_optimizer, *,
                   reconstruction_weight=500.0, reuse_generator=False,
                   share_history=True,
                   measure=None, observe=None, precision="fp32",
                   discriminator_steps=1, generator_steps=1, return_terms=False):
    """D then G, with unchanged BCE targets and reconstruction reduction.

    ``measure(name)`` is an optional context manager factory for benchmarks.
    ``observe`` is only used by equivalence tests; normal training keeps no trace.
    ``reuse_generator`` is retained for caller compatibility but ignored: D and
    G always receive independent generator forwards (and independent masks).
    ``discriminator_steps`` and ``generator_steps`` are optimizer steps on this
    batch: all D steps, then all G steps, each with its own complete forward and
    backward. The returned losses are means over the steps of each network.
    ``return_terms`` adds a third value: the unweighted reconstruction error of G
    (the masked MSE, comparable between runs with different weights), the addends
    of the two losses (their sum is the loss: weighted reconstruction + adversarial
    for G, half the real and half the fake BCE for D) and the mean D output on real
    and on generated data in the D step. They are copies of values the step
    computes anyway. D outputs are anomaly probabilities: the target is 0 for real
    and 1 for generated.
    """
    if min(discriminator_steps, generator_steps) < 1:
        raise ValueError("discriminator_steps and generator_steps must be positive.")
    scope = measure if measure is not None else lambda name: nullcontext()
    def emit(name, **values):
        if observe is not None:
            observe(name, values)
    recent, trend, mask, calendar, observed = batch
    # Separate autocast scopes around forwards; backward and updates remain outside.
    def amp():
        return autocast_context(precision, recent.device)
    logits_options = {"return_logits": True} if precision == "bf16" else {}
    adversarial_loss = binary_cross_entropy_with_logits if precision == "bf16" else binary_cross_entropy
    probability = (lambda output: torch.sigmoid(output.detach().float())) if precision == "bf16" else (
        lambda output: output.detach().float())
    normal = fake_target = None
    generator_optimizer.zero_grad()
    discriminator_losses, generator_losses = [], []
    discriminator_terms, generator_terms = [], []
    for _ in range(discriminator_steps):
        discriminator_optimizer.zero_grad()
        with scope("G_forward"), amp():
            with torch.no_grad():
                generated = model.generator(recent, trend, mask, calendar).float()
        with scope("D_forward"), amp():
            if share_history:
                real, fake = model.discriminator.score_pair(recent, observed, generated.detach(), mask, **logits_options)
            else:
                real = model.discriminator(torch.cat((recent, observed[:, None]), dim=1), mask, **logits_options)
                fake = model.discriminator(torch.cat((recent, generated.detach()[:, None]), dim=1), mask, **logits_options)
        emit("D_forward", generated=generated, real=real, fake=fake)
        if normal is None:
            # One target per output of D: a sample (patch model) or a cell of the field (full-grid model).
            normal, fake_target = torch.zeros_like(real), torch.ones_like(real)
        with scope("D_loss_check"):
            real_loss, fake_loss = adversarial_loss(real, normal), adversarial_loss(fake, fake_target)
            discriminator_loss = .5 * (real_loss + fake_loss)
            if not torch.isfinite(discriminator_loss):
                raise FloatingPointError("Non-finite discriminator loss; stopping before exporting scores.")
        with scope("D_backward"):
            discriminator_loss.backward()
        emit("D_backward", model=model, loss=discriminator_loss)
        with scope("D_optimizer"):
            discriminator_optimizer.step()
        emit("D_step", model=model)
        discriminator_losses.append(discriminator_loss.detach())
        if return_terms:
            discriminator_terms.append(torch.stack((.5 * real_loss.detach(), .5 * fake_loss.detach(),
                                                    probability(real).mean(), probability(fake).mean())))
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        for _ in range(generator_steps):
            generator_optimizer.zero_grad()
            with scope("G_forward"), amp():
                generated = model.generator(recent, trend, mask, calendar).float()
            with scope("D_forward_for_G"), amp():
                # This forward deliberately recomputes history with UPDATED D weights.
                if share_history:
                    # Keep constant history independent of generated: concatenating them
                    # would build a useless backward path through all history convolutions.
                    history = model.discriminator.encode_history(recent, mask)
                    fake = model.discriminator.score_current(history, generated, mask, **logits_options)
                else:
                    fake = model.discriminator(torch.cat((recent, generated[:, None]), dim=1), mask, **logits_options)
            emit("G_forward", generated=generated, fake=fake)
            with scope("G_loss_check"):
                errors = torch.where(mask.bool(), generated - observed, 0.0).square()
                reconstruction_error = masked_cell_mean(errors, mask).mean()
                reconstruction_loss = reconstruction_weight * reconstruction_error
                adversarial = adversarial_loss(fake, normal)
                generator_loss = reconstruction_loss + adversarial
                if not torch.isfinite(generator_loss):
                    raise FloatingPointError("Non-finite generator loss; stopping before exporting scores.")
            with scope("G_backward"):
                generator_loss.backward()
            emit("G_backward", model=model, loss=generator_loss)
            with scope("G_optimizer"):
                generator_optimizer.step()
            emit("G_step", model=model)
            generator_losses.append(generator_loss.detach())
            if return_terms:
                generator_terms.append(torch.stack((reconstruction_error.detach(), reconstruction_loss.detach(),
                                                    adversarial.detach())))
    finally:
        for parameter in model.discriminator.parameters():
            parameter.requires_grad_(True)
    losses = _mean_loss(generator_losses), _mean_loss(discriminator_losses)
    return (*losses, _terms(generator_terms, discriminator_terms)) if return_terms else losses
