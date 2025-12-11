"""Simulated scenarios for PACD/IVAPCI benchmarks.

This module implements the simulation interface described in
``docs/pacd_benchmark_design.md``. It provides several synthetic
scenarios that control confounding strength, nonlinearity, and proxy
quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    d_u: int
    d_x: int
    d_w: int
    d_z: int
    gamma: float
    nonlinear: bool = False
    strong_nonlinear: bool = False
    weak_overlap: bool = False
    hetero_tau: bool = False
    misaligned_proxies: bool = False
    sparse_signal: bool = False
    heavy_tail_latent: bool = False
    mixture_latent: bool = False


SCENARIOS: Dict[str, ScenarioSpec] = {
    "EASY-linear-weak": ScenarioSpec(
        name="EASY-linear-weak", d_u=2, d_x=6, d_w=6, d_z=6, gamma=0.6
    ),
    "EASY-linear-strong": ScenarioSpec(
        name="EASY-linear-strong", d_u=2, d_x=8, d_w=8, d_z=8, gamma=1.2
    ),
    "MODERATE-nonlinear": ScenarioSpec(
        name="MODERATE-nonlinear",
        d_u=3,
        d_x=8,
        d_w=6,
        d_z=6,
        gamma=0.9,
        nonlinear=True,
    ),
    "HARD-nonlinear-strong": ScenarioSpec(
        name="HARD-nonlinear-strong",
        d_u=3,
        d_x=10,
        d_w=8,
        d_z=8,
        gamma=1.4,
        nonlinear=True,
        strong_nonlinear=True,
    ),
    "HARD-nonlinear-extreme": ScenarioSpec(
        name="HARD-nonlinear-extreme",
        d_u=4,
        d_x=14,
        d_w=10,
        d_z=10,
        gamma=2.0,
        nonlinear=True,
        strong_nonlinear=True,
        weak_overlap=True,
        hetero_tau=True,
        misaligned_proxies=True,
        sparse_signal=True,
    ),
    "HARD-nonlinear-extreme-heavy-tail": ScenarioSpec(
        name="HARD-nonlinear-extreme-heavy-tail",
        d_u=4,
        d_x=14,
        d_w=10,
        d_z=10,
        gamma=2.0,
        nonlinear=True,
        strong_nonlinear=True,
        weak_overlap=True,
        hetero_tau=True,
        misaligned_proxies=True,
        sparse_signal=True,
        heavy_tail_latent=True,
    ),
    "HARD-nonlinear-extreme-mixture": ScenarioSpec(
        name="HARD-nonlinear-extreme-mixture",
        d_u=4,
        d_x=14,
        d_w=10,
        d_z=10,
        gamma=2.0,
        nonlinear=True,
        strong_nonlinear=True,
        weak_overlap=True,
        hetero_tau=True,
        misaligned_proxies=True,
        sparse_signal=True,
        mixture_latent=True,
    ),
    "HARD-nonlinear-weak-overlap": ScenarioSpec(
        name="HARD-nonlinear-weak-overlap",
        d_u=3,
        d_x=10,
        d_w=8,
        d_z=8,
        gamma=1.4,
        nonlinear=True,
        strong_nonlinear=True,
        weak_overlap=True,
    ),
    "HARD-nonlinear-hetero-tau": ScenarioSpec(
        name="HARD-nonlinear-hetero-tau",
        d_u=3,
        d_x=10,
        d_w=8,
        d_z=8,
        gamma=1.4,
        nonlinear=True,
        strong_nonlinear=True,
        hetero_tau=True,
    ),
    "HARD-nonlinear-misaligned-proxies": ScenarioSpec(
        name="HARD-nonlinear-misaligned-proxies",
        d_u=3,
        d_x=10,
        d_w=8,
        d_z=8,
        gamma=1.4,
        nonlinear=True,
        strong_nonlinear=True,
        misaligned_proxies=True,
        sparse_signal=True,
    ),
}


def list_scenarios() -> Dict[str, ScenarioSpec]:
    """Return available simulation scenario specifications."""

    return SCENARIOS.copy()


def simulate_scenario(
    scenario: str, n: int, seed: int, variant: str = "full_proxies"
) -> Dict[str, Any]:
    """Simulate a synthetic dataset for the requested scenario.

    Parameters
    ----------
    scenario:
        Scenario name. Options are keys of :data:`SCENARIOS`.
    n:
        Sample size.
    seed:
        Random seed for reproducibility.
    variant:
        Proxy/observability variant. Supported values:
        ``"full_proxies"``, ``"weak_proxies"``, ``"missing_Z"``,
        ``"missing_W"``, ``"partial_X"``.

    Returns
    -------
    dict
        A dictionary containing simulated arrays and metadata. Keys:
        ``X, W, Z, A, Y, U, tau, meta``.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(SCENARIOS)}")

    rng = np.random.default_rng(seed)
    spec = SCENARIOS[scenario]

    U = _sample_latent(n, spec, rng)

    X, W, Z = _generate_proxies(U, spec, rng, variant)

    logits_A = _treatment_logit(U, X, Z, spec, rng)
    A = rng.binomial(1, _sigmoid(logits_A))

    Y0, Y1 = _potential_outcomes(U, X, W, spec, rng)
    Y = A * Y1 + (1 - A) * Y0

    X_out, W_out, Z_out = _apply_variant_masks(X, W, Z, variant, rng)

    tau_true = float(np.mean(Y1 - Y0))

    return {
        "X": X_out,
        "W": W_out,
        "Z": Z_out,
        "A": A,
        "Y": Y,
        "U": U,
        "tau": tau_true,
        "meta": {
            "scenario": scenario,
            "variant": variant,
            "seed": seed,
            "spec": spec,
        },
    }


def _generate_proxies(
    U: np.ndarray, spec: ScenarioSpec, rng: np.random.Generator, variant: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal_scale = 1.0
    noise_scale = 1.0
    if variant == "weak_proxies":
        signal_scale = 0.35
        noise_scale = 2.2
    elif variant not in {"full_proxies", "missing_Z", "missing_W", "partial_X"}:
        raise ValueError(f"Unsupported variant '{variant}'.")

    d_u = spec.d_u
    # base weights
    X_weights = rng.normal(scale=signal_scale, size=(d_u, spec.d_x))
    W_weights = rng.normal(scale=signal_scale, size=(d_u, spec.d_w))
    Z_weights = rng.normal(scale=signal_scale, size=(d_u, spec.d_z))

    X_noise = rng.normal(scale=noise_scale, size=(U.shape[0], spec.d_x))
    W_noise = rng.normal(scale=noise_scale, size=(U.shape[0], spec.d_w))
    Z_noise = rng.normal(scale=noise_scale, size=(U.shape[0], spec.d_z))

    if spec.misaligned_proxies:
        # X leans toward treatment-related signals, W leans toward outcome-related signals.
        treat_weights = rng.normal(scale=signal_scale * 1.2, size=(d_u, spec.d_x))
        outcome_weights = rng.normal(scale=signal_scale * 1.2, size=(d_u, spec.d_w))
        X = U @ treat_weights + X_noise
        W = U @ outcome_weights + W_noise
        # Z mixes a few treatment-aligned dimensions with mostly noisy ones
        z_signal_dims = max(1, spec.d_z // 3)
        Z = U @ Z_weights
        Z[:, z_signal_dims:] = rng.normal(scale=noise_scale, size=(U.shape[0], spec.d_z - z_signal_dims))
        Z += Z_noise
    else:
        X = U @ X_weights + X_noise
        W = U @ W_weights + W_noise
        Z = U @ Z_weights + Z_noise

    if spec.sparse_signal:
        # Only a subset of dimensions carry confounding signal; remaining dimensions are pure noise.
        sig_x = max(1, spec.d_x // 3)
        sig_w = max(1, spec.d_w // 3)
        sig_z = max(1, spec.d_z // 3)
        X[:, sig_x:] = rng.normal(scale=noise_scale, size=(U.shape[0], spec.d_x - sig_x))
        W[:, sig_w:] = rng.normal(scale=noise_scale, size=(U.shape[0], spec.d_w - sig_w))
        Z[:, sig_z:] = rng.normal(scale=noise_scale, size=(U.shape[0], spec.d_z - sig_z))

    if spec.nonlinear:
        X += 0.3 * np.sin(U @ rng.normal(size=(d_u, spec.d_x)))
        W += 0.3 * np.tanh(U @ rng.normal(size=(d_u, spec.d_w)))
        Z += 0.3 * np.sin(U @ rng.normal(size=(d_u, spec.d_z)))

    return X, W, Z


def _sample_latent(n: int, spec: ScenarioSpec, rng: np.random.Generator) -> np.ndarray:
    """Sample latent confounders with optional mixture or heavy-tail structures."""

    if spec.mixture_latent:
        centers = np.array([[1.0] * spec.d_u, [-1.0] * spec.d_u])
        assignments = rng.integers(0, 2, size=n)
        U = centers[assignments] + rng.normal(scale=0.5, size=(n, spec.d_u))
    elif spec.heavy_tail_latent:
        U = rng.standard_t(df=3.0, size=(n, spec.d_u))
    else:
        U = rng.normal(size=(n, spec.d_u))
    return U


def _treatment_logit(
    U: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    spec: ScenarioSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    base = spec.gamma * (U @ rng.normal(size=(spec.d_u,)) + 0.3 * np.sum(U**2, axis=1))
    z_part = 0.5 * np.sum(Z[:, : min(3, spec.d_z)], axis=1)
    x_part = 0.15 * np.sum(X[:, : min(5, spec.d_x)], axis=1)

    logits = base + z_part + x_part

    if spec.nonlinear:
        logits += 0.3 * np.sin(z_part) + 0.2 * np.tanh(x_part)
    if spec.strong_nonlinear:
        logits += 0.25 * np.sin(np.sum(U, axis=1) * np.sum(Z, axis=1))

    # Weak-overlap branch: deliberately push e close to 0 or 1.
    if spec.weak_overlap:
        gate = np.sum(U[:, : min(2, spec.d_u)] * Z[:, : min(2, spec.d_z)], axis=1)
        logits += 2.5 * np.tanh(gate)

    return logits


def _potential_outcomes(
    U: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    spec: ScenarioSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    base_signal = spec.gamma * (U @ rng.normal(size=(spec.d_u,)) + 0.5 * np.sum(U, axis=1))
    x_part = 0.2 * np.sum(X[:, : min(4, spec.d_x)], axis=1)
    w_part = 0.25 * np.sum(W[:, : min(4, spec.d_w)], axis=1)

    if spec.nonlinear:
        x_part += 0.2 * np.sin(x_part)
        w_part += 0.2 * np.tanh(w_part)
    if spec.strong_nonlinear:
        cross = np.sum(U * W[:, : spec.d_u], axis=1)
        x_part += 0.2 * np.sin(cross)
        w_part += 0.1 * (X[:, 0] * W[:, 0])

    treatment_effect = 1.5 if spec.strong_nonlinear else 1.0

    if spec.hetero_tau:
        hetero_tau = 0.6 * np.sin(U[:, 0] + 0.5 * (U[:, 1] ** 2 if U.shape[1] > 1 else 0.0))
        treatment_effect = treatment_effect + hetero_tau

    noise0 = rng.normal(scale=1.0, size=U.shape[0])
    noise1 = rng.normal(scale=1.0, size=U.shape[0])

    Y0 = base_signal + x_part + 0.5 * w_part + noise0
    Y1 = base_signal + x_part + 0.5 * w_part + treatment_effect + 0.8 * w_part + noise1

    return Y0, Y1


def _apply_variant_masks(
    X: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    variant: str,
    rng: np.random.Generator,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    X_out: Optional[np.ndarray] = X
    W_out: Optional[np.ndarray] = W
    Z_out: Optional[np.ndarray] = Z

    if variant == "missing_Z":
        Z_out = None
    elif variant == "missing_W":
        W_out = None
    elif variant == "partial_X":
        keep = max(1, X.shape[1] // 2)
        columns = np.sort(rng.choice(X.shape[1], size=keep, replace=False))
        X_out = X[:, columns]

    return X_out, W_out, Z_out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
