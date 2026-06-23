"""Training losses for the PVGIS ST-GNN run.

The PV prediction path is trained with the heteroscedastic Gaussian negative
log-likelihood of SDE-Net's regression experiment (Kong et al., 2020, supp.
S.4.2): ``log(sigma^2) + (y - mean)^2 / sigma^2``.  This is the drift net's
aleatoric-uncertainty objective; the auxiliary irradiance (kt) head keeps a
plain MSE.  Both are separate from SDE-Net's BCE diffusion objective, which
lives in the training loop.

A heavier-tailed alternative to the Gaussian path is provided by
``student_t_nll`` (location-scale Student-t): it keeps the same (mean, sigma)
head and the same beta-NLL weighting, but a finite ``nu`` puts more predictive
mass in the tails (better extreme-event coverage). ``student_t_ppf`` supplies
the matching predictive-interval quantile without a SciPy dependency.
"""
from __future__ import annotations

import math

import torch


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    beta: float = 0.0,
) -> torch.Tensor:
    """Element-wise heteroscedastic Gaussian NLL (Kong et al. regression form):
    ``log(sigma^2) + (target - mean)^2 / sigma^2`` (no reduction; ``sigma > 0``).

    ``beta`` adds the beta-NLL weighting of Seitzer et al. (2022): each element
    is scaled by ``stopgrad(sigma)^(2*beta)``. ``beta=0`` is the plain NLL;
    ``beta=0.5`` restores MSE-like gradients on the mean (no variance runaway).
    """
    nll = torch.log(sigma ** 2) + (target - mean) ** 2 / (sigma ** 2)
    if beta > 0.0:
        nll = sigma.detach() ** (2.0 * beta) * nll
    return nll


def student_t_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    nu: float,
    beta: float = 0.0,
) -> torch.Tensor:
    """Element-wise heteroscedastic location-scale Student-t NLL (no reduction).

    ``sigma`` is the t *scale* (the same softplus head output the Gaussian path
    uses), ``nu`` the fixed degrees of freedom (``> 0``; ``> 2`` for finite
    variance). Written in the Kong "doubled" convention (no 1/2 factor) so that
    as ``nu -> inf`` it reduces to :func:`gaussian_nll` up to an additive
    constant; a small ``nu`` gives heavier tails than the Gaussian, i.e. more
    predictive mass on extreme residuals.

    ``beta`` applies the same Seitzer et al. (2022) beta-NLL weighting as
    :func:`gaussian_nll`: each element is scaled by ``stopgrad(sigma)^(2*beta)``.
    """
    nu_t = torch.as_tensor(float(nu), dtype=sigma.dtype, device=sigma.device)
    z2 = ((target - mean) / sigma) ** 2
    const = 2.0 * (torch.lgamma(nu_t / 2.0) - torch.lgamma((nu_t + 1.0) / 2.0)) \
        + torch.log(nu_t * math.pi)
    nll = const + torch.log(sigma ** 2) + (nu_t + 1.0) * torch.log1p(z2 / nu_t)
    if beta > 0.0:
        nll = sigma.detach() ** (2.0 * beta) * nll
    return nll


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 300, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_ppf(p: float, nu: float) -> float:
    """Inverse CDF (quantile) of the standard Student-t with ``nu`` dof.

    Dependency-free (no SciPy): the t-CDF is evaluated via the regularised
    incomplete beta and inverted by bisection. Used to size predictive
    intervals so a Student-t-trained model is scored with t-quantiles rather
    than the Gaussian 1.96 (which would under-cover the heavy tails).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}.")
    if p < 0.5:
        return -student_t_ppf(1.0 - p, nu)
    nu = float(nu)

    def tcdf(t: float) -> float:
        xb = nu / (nu + t * t)
        ib = _betai(nu / 2.0, 0.5, xb)
        return 1.0 - 0.5 * ib if t > 0.0 else 0.5 * ib

    lo, hi = 0.0, 1.0e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tcdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_loss_fn(reduction: str = "mean") -> torch.nn.Module:
    """Build the auxiliary point-loss module: torch.nn.MSELoss().

    Used for the optional irradiance (kt) head only. The PV prediction path uses
    ``gaussian_nll`` instead; neither replaces SDE-Net's BCE diffusion objective.
    """
    return torch.nn.MSELoss(reduction=reduction)
