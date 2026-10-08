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
    if getattr(model, "global_graph", False):
        return graph_gan_train_step(model, batch, generator_optimizer, discriminator_optimizer,
            reconstruction_weight=reconstruction_weight, share_history=share_history,
            measure=measure, observe=observe, precision=precision,
            discriminator_steps=discriminator_steps, generator_steps=generator_steps,
            return_terms=return_terms)
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
    normal = torch.zeros((recent.shape[0], 1), device=recent.device)
    fake_target = torch.ones_like(normal)
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


def graph_gan_train_step(model, batch, generator_optimizer, discriminator_optimizer, *,
                         reconstruction_weight=500., share_history=True, measure=None, observe=None, precision="fp32",
                         discriminator_steps=1, generator_steps=1, return_terms=False):
    """Same D/G objectives over all centers; chunk D activations, two global G calls.

    Accumulate dL/d(prediction) on a detached leaf, then backpropagate once through
    G. This is the chain rule for the unchunked mean, without retaining D graphs.
    Chunks only bound memory: their gradients accumulate into ONE optimizer step.
    ``discriminator_steps``/``generator_steps`` repeat that complete update.

    The history and observed patches depend on the batch only, not on any weight:
    they are gathered once and serve every D and G step. The patches of the generated
    values are gathered again from each generator forward. A non-finite chunk loss is
    recorded on the device and raised once per step, before the optimizer step: no
    host read per chunk.
    """
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
    count = recent.shape[0] * recent.shape[2]
    chunks = list(model.patch_inputs(recent, observed))
    generator_optimizer.zero_grad()
    discriminator_losses, generator_losses = [], []
    discriminator_terms, generator_terms = [], []
    for _ in range(discriminator_steps):
        discriminator_optimizer.zero_grad()
        with scope("G_forward"), amp(), torch.no_grad():
            generated = model.generator(recent, trend, mask, calendar).float()
        discriminator_loss, step_terms = recent.new_zeros(()), recent.new_zeros(4)
        finite = torch.ones((), dtype=torch.bool, device=recent.device)
        for ids, times, safe, valid, history, real_patch in chunks:
            fake_patch = model.gather_patch(generated, times, safe, valid)
            with scope("D_forward"), amp():
                real, fake = model.discriminator.score_pair(history, real_patch, fake_patch, valid, **logits_options) if share_history else (
                    model.discriminator(torch.cat((history, real_patch[:, None]), 1), valid, **logits_options),
                    model.discriminator(torch.cat((history, fake_patch[:, None]), 1), valid, **logits_options))
                real_loss = adversarial_loss(real, torch.zeros_like(real))
                fake_loss = adversarial_loss(fake, torch.ones_like(fake))
                loss = .5 * (real_loss + fake_loss) * (len(ids) / count)
            finite = finite & torch.isfinite(loss.detach())
            with scope("D_backward"):
                loss.backward()
            discriminator_loss += loss.detach()
            if return_terms:  # Chunk-weighted, like the loss they add up to.
                weight = len(ids) / count
                step_terms += torch.stack((.5 * real_loss.detach() * weight, .5 * fake_loss.detach() * weight,
                                           probability(real).sum() / count, probability(fake).sum() / count))
        if not finite:  # Any chunk: checked once, before the weights are touched.
            raise FloatingPointError("Non-finite discriminator loss.")
        emit("D_forward", generated=generated)
        emit("D_backward", model=model, loss=discriminator_loss)
        with scope("D_optimizer"):
            discriminator_optimizer.step()
        emit("D_step", model=model)
        discriminator_losses.append(discriminator_loss)
        if return_terms:
            discriminator_terms.append(step_terms)
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        for _ in range(generator_steps):
            generator_optimizer.zero_grad()
            with scope("G_forward"), amp():
                generated = model.generator(recent, trend, mask, calendar).float()
            prediction_leaf = generated.detach().requires_grad_(True)
            generator_loss, step_terms = recent.new_zeros(()), recent.new_zeros(3)
            finite = torch.ones((), dtype=torch.bool, device=recent.device)
            for ids, times, safe, valid, history, real_patch in chunks:
                fake_patch = model.gather_patch(prediction_leaf, times, safe, valid)
                with scope("D_forward_for_G"), amp():
                    fake = (model.discriminator.score_current(model.discriminator.encode_history(history, valid), fake_patch, valid, **logits_options)
                            if share_history else model.discriminator(torch.cat((history, fake_patch[:, None]), 1), valid, **logits_options))
                    errors = (fake_patch - real_patch).square()
                    reconstruction_error = masked_cell_mean(errors, valid).mean()
                    reconstruction_loss = reconstruction_weight * reconstruction_error
                    adversarial = adversarial_loss(fake, torch.zeros_like(fake))
                    loss = (reconstruction_loss + adversarial) * (len(ids) / count)
                finite = finite & torch.isfinite(loss.detach())
                with scope("G_backward"):
                    loss.backward()
                generator_loss += loss.detach()
                if return_terms:
                    step_terms += torch.stack((reconstruction_error.detach(), reconstruction_loss.detach(),
                                               adversarial.detach())) * (len(ids) / count)
            if not finite:  # Any chunk: checked once, before G's backward and its update.
                raise FloatingPointError("Non-finite generator loss.")
            emit("G_forward", generated=generated)
            with scope("G_backward"):
                generated.backward(prediction_leaf.grad)
            emit("G_backward", model=model, loss=generator_loss)
            with scope("G_optimizer"):
                generator_optimizer.step()
            emit("G_step", model=model)
            generator_losses.append(generator_loss)
            if return_terms:
                generator_terms.append(step_terms)
    finally:
        for parameter in model.discriminator.parameters():
            parameter.requires_grad_(True)
    losses = _mean_loss(generator_losses), _mean_loss(discriminator_losses)
    return (*losses, _terms(generator_terms, discriminator_terms)) if return_terms else losses
