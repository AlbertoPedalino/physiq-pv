"""Equivalent STGAN alternating updates, with optional diagnostic scopes."""
from contextlib import nullcontext
import torch
from torch.nn.functional import binary_cross_entropy
from .model import masked_cell_mean


class DeviceLossTotals:
    """Match the old float64 weighted accumulation without per-step host reads."""
    def __init__(self, device):
        self.totals = torch.zeros(2, dtype=torch.float64, device=device)
        self.samples = 0
        self._logged_totals = [0.0, 0.0]
        self._logged_samples = 0

    def update(self, generator_loss, discriminator_loss, count):
        self.totals.add_(torch.stack((generator_loss.detach(), discriminator_loss.detach())).to(torch.float64) * count)
        self.samples += count

    def means(self):
        return (self.totals / self.samples).cpu().tolist()

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


def gan_train_step(model, batch, generator_optimizer, discriminator_optimizer, *,
                   reconstruction_weight=500.0, reuse_generator=False,
                   share_history=True,
                   measure=None, observe=None):
    """D then G, with unchanged BCE targets and reconstruction reduction.

    ``measure(name)`` is an optional context manager factory for benchmarks.
    ``observe`` is only used by equivalence tests; normal training keeps no trace.
    ``reuse_generator`` is retained for caller compatibility but ignored: D and
    G always receive independent generator forwards (and independent masks).
    """
    if getattr(model, "global_graph", False):
        return graph_gan_train_step(model, batch, generator_optimizer, discriminator_optimizer,
            reconstruction_weight=reconstruction_weight, share_history=share_history,
            measure=measure, observe=observe)
    scope = measure if measure is not None else lambda name: nullcontext()
    def emit(name, **values):
        if observe is not None:
            observe(name, values)
    recent, trend, mask, calendar, observed = batch
    normal = torch.zeros((recent.shape[0], 1), device=recent.device)
    fake_target = torch.ones_like(normal)
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    with scope("G_forward"):
        with torch.no_grad():
            generated = model.generator(recent, trend, mask, calendar)
    with scope("D_forward"):
        if share_history:
            real, fake = model.discriminator.score_pair(recent, observed, generated.detach(), mask)
        else:
            real = model.discriminator(torch.cat((recent, observed[:, None]), dim=1), mask)
            fake = model.discriminator(torch.cat((recent, generated.detach()[:, None]), dim=1), mask)
    emit("D_forward", generated=generated, real=real, fake=fake)
    with scope("D_loss_check"):
        discriminator_loss = .5 * (binary_cross_entropy(real, normal) + binary_cross_entropy(fake, fake_target))
        if not torch.isfinite(discriminator_loss):
            raise FloatingPointError("Non-finite discriminator loss; stopping before exporting scores.")
    with scope("D_backward"):
        discriminator_loss.backward()
    emit("D_backward", model=model, loss=discriminator_loss)
    with scope("D_optimizer"):
        discriminator_optimizer.step()
    emit("D_step", model=model)
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        with scope("G_forward"):
            generated = model.generator(recent, trend, mask, calendar)
        with scope("D_forward_for_G"):
            # This forward deliberately recomputes history with UPDATED D weights.
            if share_history:
                # Keep constant history independent of generated: concatenating them
                # would build a useless backward path through all history convolutions.
                history = model.discriminator.encode_history(recent, mask)
                fake = model.discriminator.score_current(history, generated, mask)
            else:
                fake = model.discriminator(torch.cat((recent, generated[:, None]), dim=1), mask)
        emit("G_forward", generated=generated, fake=fake)
        with scope("G_loss_check"):
            errors = torch.where(mask.bool(), generated - observed, 0.0).square()
            generator_loss = reconstruction_weight * masked_cell_mean(errors, mask).mean() + binary_cross_entropy(fake, normal)
            if not torch.isfinite(generator_loss):
                raise FloatingPointError("Non-finite generator loss; stopping before exporting scores.")
        with scope("G_backward"):
            generator_loss.backward()
        emit("G_backward", model=model, loss=generator_loss)
        with scope("G_optimizer"):
            generator_optimizer.step()
        emit("G_step", model=model)
    finally:
        for parameter in model.discriminator.parameters():
            parameter.requires_grad_(True)
    return generator_loss.detach(), discriminator_loss.detach()


def graph_gan_train_step(model, batch, generator_optimizer, discriminator_optimizer, *,
                         reconstruction_weight=500., share_history=True, measure=None, observe=None):
    """Same D/G objectives over all centers; chunk D activations, two global G calls.

    Accumulate dL/d(prediction) on a detached leaf, then backpropagate once through
    G. This is the chain rule for the unchunked mean, without retaining D graphs.
    """
    scope = measure if measure is not None else lambda name: nullcontext()
    def emit(name, **values):
        if observe is not None:
            observe(name, values)
    recent, trend, mask, calendar, observed = batch
    count = recent.shape[0] * recent.shape[2]
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    with scope("G_forward"), torch.no_grad():
        generated = model.generator(recent, trend, mask, calendar)
    discriminator_loss = recent.new_zeros(())
    for ids, history, real_patch, fake_patch, valid in model.patch_batches(recent, observed, generated):
        with scope("D_forward"):
            real, fake = model.discriminator.score_pair(history, real_patch, fake_patch, valid) if share_history else (
                model.discriminator(torch.cat((history, real_patch[:, None]), 1), valid),
                model.discriminator(torch.cat((history, fake_patch[:, None]), 1), valid))
            loss = .5 * (binary_cross_entropy(real, torch.zeros_like(real))
                         + binary_cross_entropy(fake, torch.ones_like(fake))) * (len(ids) / count)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite discriminator loss.")
        with scope("D_backward"):
            loss.backward()
        discriminator_loss += loss.detach()
    emit("D_forward", generated=generated)
    emit("D_backward", model=model, loss=discriminator_loss)
    with scope("D_optimizer"):
        discriminator_optimizer.step()
    emit("D_step", model=model)
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        with scope("G_forward"):
            generated = model.generator(recent, trend, mask, calendar)
        prediction_leaf = generated.detach().requires_grad_(True)
        generator_loss = recent.new_zeros(())
        for ids, history, real_patch, fake_patch, valid in model.patch_batches(recent, observed, prediction_leaf):
            with scope("D_forward_for_G"):
                fake = (model.discriminator.score_current(model.discriminator.encode_history(history, valid), fake_patch, valid)
                        if share_history else model.discriminator(torch.cat((history, fake_patch[:, None]), 1), valid))
                errors = (fake_patch - real_patch).square()
                loss = (reconstruction_weight * masked_cell_mean(errors, valid).mean()
                        + binary_cross_entropy(fake, torch.zeros_like(fake))) * (len(ids) / count)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite generator loss.")
            with scope("G_backward"):
                loss.backward()
            generator_loss += loss.detach()
        emit("G_forward", generated=generated)
        with scope("G_backward"):
            generated.backward(prediction_leaf.grad)
        emit("G_backward", model=model, loss=generator_loss)
        with scope("G_optimizer"):
            generator_optimizer.step()
        emit("G_step", model=model)
    finally:
        for parameter in model.discriminator.parameters():
            parameter.requires_grad_(True)
    return generator_loss, discriminator_loss
