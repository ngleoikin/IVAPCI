"""IVAPCI v3.3 (TheoryComplete): hierarchical encoder + theorem-aware extras."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from . import BaseCausalEstimator
from .ivapci_theory_diagnostics import (
    TheoremComplianceDiagnostics,
    TheoremDiagnosticsConfig,
)


# -------------------------
# Theorem 1: preprocessing
# -------------------------

def _info_aware_standardize(
    train: np.ndarray,
    min_std: float = 1e-2,
    low_var_min_std: float = 1e-4,
    clip_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize while *clamping* (not flattening) near-constant dimensions.

    Near-deterministic proxy channels keep their small variance (plus a tiny
    floor) instead of being inflated to the global ``min_std``. This protects
    low-variance informative proxies in "low-var" scenarios from being washed
    out by preprocessing.

    Returns:
        train_std, mean, std_clamped, protected_mask
    """
    train = np.asarray(train, dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True, dtype=np.float32)
    std = train.std(axis=0, keepdims=True, dtype=np.float32)
    protected = (std < min_std).squeeze(0)

    # Preserve true (small) std for low-variance channels; only apply a tiny floor to avoid divide-by-zero.
    std_clamped = np.where(std < low_var_min_std, np.maximum(std, low_var_min_std), std).astype(np.float32)
    standardized = ((train - mean) / std_clamped).astype(np.float32)
    if clip_value is not None:
        standardized = np.clip(standardized, -clip_value, clip_value).astype(np.float32)
    return standardized, mean.astype(np.float32), std_clamped, protected


def _effective_sample_size(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    num = float(np.sum(weights) ** 2)
    den = float(np.sum(weights**2) + 1e-12)
    return num / den if den > 0 else 0.0


def _compute_residual(target: torch.Tensor, condition: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Project out ``condition`` from ``target`` to approximate conditional adversaries."""

    condition_c = condition - condition.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)

    ct_c = condition_c.T @ condition_c / max(1, condition_c.shape[0])
    i_eye = torch.eye(ct_c.shape[0], device=condition.device, dtype=condition.dtype)
    ct_t = condition_c.T @ target_c / max(1, condition_c.shape[0])
    try:
        beta = torch.linalg.solve(ct_c + ridge * i_eye, ct_t)
    except RuntimeError:
        beta = torch.zeros(condition_c.shape[1], target_c.shape[1], device=condition.device, dtype=condition.dtype)
    target_res = target_c - condition_c @ beta
    return target_res


class TrueConditionalAdversary(nn.Module):
    """Conditional adversary via concat+detach with configurable loss type."""

    def __init__(self, target_dim: int, condition_dim: int, hidden: Sequence[int], *, loss: str = "bce"):
        super().__init__()
        self.discriminator = _mlp(target_dim + condition_dim, hidden, 1)
        loss = loss.lower()
        if loss not in {"bce", "mse"}:
            raise ValueError("loss must be 'bce' or 'mse'")
        self.loss = loss

    def forward(self, target: torch.Tensor, condition: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([target, condition.detach()], dim=1)
        logits = self.discriminator(combined).squeeze(-1)
        if self.loss == "bce":
            return nn.functional.binary_cross_entropy_with_logits(logits, label)
        return nn.functional.mse_loss(logits, label)


def compute_ess_by_group(e_hat: np.ndarray, A: np.ndarray) -> Tuple[float, float]:
    eps = 1e-8
    treated_mask = A == 1
    control_mask = A == 0

    if treated_mask.sum() > 0:
        w_treated = 1.0 / np.clip(e_hat[treated_mask], eps, 1 - eps)
        ess_treated = _effective_sample_size(w_treated) / treated_mask.sum()
    else:
        ess_treated = 0.0

    if control_mask.sum() > 0:
        w_control = 1.0 / np.clip(1 - e_hat[control_mask], eps, 1 - eps)
        ess_control = _effective_sample_size(w_control) / control_mask.sum()
    else:
        ess_control = 0.0

    return ess_treated, ess_control


def find_optimal_clip_threshold(
    e_raw: np.ndarray,
    A: np.ndarray,
    ess_target: float,
    min_clip: float,
    max_clip: float,
    n_search: int = 20,
) -> Tuple[float, dict]:
    candidates = np.linspace(min_clip, max_clip, n_search)
    best_clip = max_clip
    best_min_ess = 0.0

    for clip_val in candidates:
        e_clipped = np.clip(e_raw, clip_val, 1 - clip_val)
        ess_t, ess_c = compute_ess_by_group(e_clipped, A)
        min_ess = min(ess_t, ess_c)
        if min_ess >= ess_target:
            best_clip = clip_val
            best_min_ess = min_ess
            break
        if min_ess > best_min_ess:
            best_clip = clip_val
            best_min_ess = min_ess

    e_final = np.clip(e_raw, best_clip, 1 - best_clip)
    ess_t, ess_c = compute_ess_by_group(e_final, A)

    eps = 1e-8
    w_ate = A / np.clip(e_final, eps, 1 - eps) + (1 - A) / np.clip(1 - e_final, eps, 1 - eps)
    overall_ess = _effective_sample_size(w_ate)
    overall_ess_ratio = overall_ess / len(A) if len(A) else 0.0

    stats = {
        "clip_threshold": float(best_clip),
        "ess_treated": float(ess_t),
        "ess_control": float(ess_c),
        "ess_min": float(min(ess_t, ess_c)),
        "ess_overall_ratio": float(overall_ess_ratio),
        "frac_clipped": float(np.mean((e_raw < best_clip) | (e_raw > 1 - best_clip))),
    }
    return float(best_clip), stats


def adaptive_ipw_cap_by_quantile(
    weights: np.ndarray,
    quantile: float,
    min_cap: float,
    max_cap: float,
) -> float:
    if weights.size == 0:
        return min_cap
    cap_candidate = float(np.quantile(np.abs(weights), quantile))
    return float(np.clip(cap_candidate, min_cap, max_cap))


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32)
    mean_f = np.asarray(mean, dtype=np.float32)
    std_f = np.asarray(std, dtype=np.float32)
    return (x_f - mean_f) / std_f


class InformationLossMonitor:
    """Reconstruction MSE / variance as a lightweight ΔI proxy (Theorem 1)."""

    @staticmethod
    def estimate(model: "IVAPCIv33TheoryHierEstimator", V_all: np.ndarray) -> Dict[str, float]:
        V_std = _apply_standardize(V_all.astype(np.float32), model._v_mean, model._v_std)
        with torch.no_grad():
            V_t = torch.from_numpy(V_std).to(model.device)
            tx, tw, tz, tn = model._encode_blocks(V_t)
            recon = model.decoder(torch.cat([tx, tw, tz, tn], dim=1)).cpu().numpy()
        recon_mse = float(np.mean((recon - V_std) ** 2))
        var = float(np.var(V_std) + 1e-8)
        return {"recon_mse": recon_mse, "info_loss_proxy": recon_mse / var}


def enhanced_training_monitor(estimator: "IVAPCIv33TheoryHierEstimator", epoch: int, losses: Dict[str, float]) -> None:
    """Lightweight training-time monitor to surface identifiability risks.

    Checks every 10 epochs:
      - W ⟂ A proxy via rep_auc_w_to_a
      - Z exclusion leakage via rep_exclusion_leakage_r2
      - IV strength via iv_relevance_abs_corr
    Optionally increases gamma_adv_w early in training when W–A dependence is high.
    """
    if epoch % 10 != 0:
        return

    diag = getattr(estimator, "training_diagnostics", {}) or {}
    warnings = []

    w_auc = float(diag.get("rep_auc_w_to_a", 0.5))
    if abs(w_auc - 0.5) > 0.1:
        warnings.append(f"⚠️  W 独立性差: AUC={w_auc:.3f}")

    z_r2 = float(diag.get("rep_exclusion_leakage_r2", 0.0))
    if z_r2 > 0.15:
        warnings.append(f"⚠️  Z 泄露: R²={z_r2:.3f}")

    iv_strength = float(diag.get("iv_relevance_abs_corr", 0.0))
    if iv_strength < 0.15:
        warnings.append(f"⚠️  弱 IV: corr={iv_strength:.3f}")

    if warnings:
        print(f"\nEpoch {epoch} 诊断 (main={losses.get('main', float('nan')):.4f}, val={losses.get('val', float('nan')):.4f}):")
        for w in warnings:
            print(f"  {w}")
        print()

    if abs(w_auc - 0.5) > 0.15 and epoch < 100:
        estimator.config.gamma_adv_w *= 1.2
        print(f"  🔧 自动调整: gamma_adv_w → {estimator.config.gamma_adv_w:.3f}")


# -------------------------
# small NN building blocks
# -------------------------

def _mlp(input_dim: int, hidden: Sequence[int], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class _GroupEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        if hidden:
            self.body = _mlp(input_dim, hidden, hidden[-1], dropout=dropout)
            self.out = nn.Linear(hidden[-1], out_dim)
        else:
            self.body = nn.Identity()
            self.out = nn.Linear(input_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.out(h)


class _Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], out_dim: int):
        super().__init__()
        self.net = _mlp(input_dim, hidden, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _AClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t).squeeze(-1)


class _YRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(input_dim + 1, hidden, 1)

    def forward(self, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, a.unsqueeze(-1)], dim=1)).squeeze(-1)


class _YAdversary(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t).squeeze(-1)


# -------------------------
# Orthogonality penalties
# -------------------------

def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def _offdiag_corr_penalty(blocks: list[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    pen = torch.zeros((), device=blocks[0].device)
    centered = [_center(b) for b in blocks]
    norms = [torch.sqrt(torch.mean(b ** 2, dim=0, keepdim=True) + eps) for b in centered]
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            bi = centered[i] / norms[i]
            bj = centered[j] / norms[j]
            corr = (bi.T @ bj) / bi.shape[0]
            pen = pen + torch.mean(corr ** 2)
    return pen


def _conditional_orthogonal_penalty(blocks: list[torch.Tensor], cond: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    pen = torch.zeros((), device=blocks[0].device)
    C = _center(cond)
    CtC = C.T @ C / C.shape[0]
    I = torch.eye(CtC.shape[0], device=C.device)
    Ct = C.T / C.shape[0]
    resid = []
    for B in blocks:
        Bc = _center(B)
        Beta = torch.linalg.solve(CtC + ridge * I, Ct @ Bc)
        resid.append(Bc - C @ Beta)
    return _offdiag_corr_penalty(resid)


def _padic_ultrametric_loss(Uc: torch.Tensor, num_triplets: int = 128) -> torch.Tensor:
    if Uc.shape[0] < 3:
        return torch.zeros((), device=Uc.device)
    n = Uc.shape[0]
    t = min(num_triplets, max(1, n // 2))
    idx = torch.randint(0, n, (t, 3), device=Uc.device)
    u_i = Uc[idx[:, 0]]
    u_j = Uc[idx[:, 1]]
    u_k = Uc[idx[:, 2]]
    d_ij = torch.norm(u_i - u_j, dim=1)
    d_jk = torch.norm(u_j - u_k, dim=1)
    d_ik = torch.norm(u_i - u_k, dim=1)
    viol = torch.relu(d_ik - torch.maximum(d_ij, d_jk))
    return torch.mean(viol ** 2)


def _rbf_hsic(x: torch.Tensor, y: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """Compute a lightweight unbiased HSIC with RBF kernels for independence penalty."""
    if x.shape[0] < 4 or y.shape[0] < 4:
        return torch.zeros((), device=x.device)
    n = x.shape[0]
    if sigma is None:
        with torch.no_grad():
            dist2 = torch.pdist(x, p=2).pow(2)
            sigma = torch.sqrt(torch.median(dist2) + 1e-6)
        sigma = float(sigma.item()) if sigma is not None else 1.0
    gamma = 1.0 / (2 * max(sigma, 1e-6) ** 2)
    Kx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
    dist2_y = torch.pdist(y, p=2).pow(2)
    sigma_y = torch.sqrt(torch.median(dist2_y) + 1e-6)
    gamma_y = 1.0 / (2 * max(float(sigma_y.item()), 1e-6) ** 2)
    Ky = torch.exp(-gamma_y * torch.cdist(y, y).pow(2))
    H = torch.eye(n, device=x.device) - torch.ones((n, n), device=x.device) / n
    Kc = H @ Kx @ H
    Lc = H @ Ky @ H
    hsic = torch.trace(Kc @ Lc) / ((n - 1) ** 2)
    return hsic


# -------------------------
# Config (Theorem 4–5 guided)
# -------------------------

@dataclass
class IVAPCIV33TheoryConfig:
    """Theory-informed default configuration balancing theorems 1–5.

    Tuned for the "v3.3" setting with emphasis on:
    - Theorem 2 prioritization (information integrity before aggressive constraints)
    - Gradual conditional orthogonality (Theorem 3A) via longer warmup
    - Moderate adversarial weights (Theorem 3B) to avoid over-penalization
    - Mildly strengthened consistency (Theorem 3C) to counteract disentanglement difficulty
    - Sample-efficiency safeguards (Theorem 4–5) through adaptive defaults
    """

    x_dim: int = 0
    w_dim: int = 0
    z_dim: int = 0

    latent_x_dim: int = 4
    latent_w_dim: int = 4
    latent_z_dim: int = 2
    latent_n_dim: int = 4

    enc_x_hidden: Sequence[int] = (128, 64)
    enc_w_hidden: Sequence[int] = (256, 128, 64)
    enc_z_hidden: Sequence[int] = (64, 32)
    enc_n_hidden: Sequence[int] = (128, 64)
    dropout_z: float = 0.2

    dec_hidden: Sequence[int] = (128, 128)

    a_hidden: Sequence[int] = (64, 32)
    y_hidden: Sequence[int] = (128, 64)
    adv_a_hidden: Sequence[int] = (64,)
    adv_y_hidden: Sequence[int] = (64,)
    # conditional adversaries (Theorem 3B)
    gamma_adv_w_cond: float = 0.12
    gamma_adv_z_cond: float = 0.1
    gamma_adv_n_cond: float = 0.08

    # optional independence penalty
    lambda_hsic: float = 0.015
    hsic_max_samples: int = 256

    lambda_recon: float = 1.0
    lambda_a: float = 0.1
    lambda_y: float = 0.5
    lambda_ortho: float = 0.01
    lambda_cond_ortho: float = 1e-3
    lambda_consistency: float = 0.08
    ridge_alpha: float = 1e-2
    standardize_nuisance: bool = True
    gamma_adv_w: float = 0.15
    gamma_adv_z: float = 0.12
    gamma_adv_n: float = 0.1
    adv_steps: int = 3
    # Multi-step adversary updates (Theorem 3B/5 practical stabilization)
    adv_steps_min: int = 1
    adv_steps_max: int = 5
    adv_steps_dynamic: bool = True  # adapt adversary steps based on leakage diagnostics
    adv_warmup_epochs: int = 10
    adv_ramp_epochs: int = 30
    gamma_padic: float = 1e-3

    min_std: float = 1e-2
    low_var_min_std: float = 1e-3
    std_clip: Optional[float] = 10.0
    use_noise_in_latent: bool = True

    lr_main: float = 1e-3
    lr_adv: float = 1.5e-3
    batch_size: int = 128
    epochs_pretrain: int = 50
    epochs_main: int = 160
    val_frac: float = 0.1
    early_stopping_patience: int = 18
    early_stopping_min_delta: float = 0.0

    n_splits_dr: int = 5
    clip_prop: float = 0.01
    clip_prop_adaptive_max: float = 0.06
    clip_prop_radr: float = 1e-2
    propensity_logreg_C: float = 0.5
    propensity_shrinkage: float = 0.02
    ipw_cap: float = 15.0
    ipw_cap_quantile: float = 0.995
    ipw_cap_high: float = 100.0
    ipw_cap_radr: Optional[float] = None
    adaptive_ipw: bool = True
    ess_target: float = 0.55

    # overlap soft-penalty (stabilize propensity logits during training)
    lambda_overlap: float = 0.02
    overlap_margin: float = 0.05
    overlap_warmup_epochs: int = 10

    # monitoring and overlap scaling hooks
    cond_adv_warmup_epochs: int = 10
    cond_adv_ramp_epochs: int = 20
    monitor_batch_size: int = 512
    monitor_every: int = 1
    monitor_ema: float = 0.8
    overlap_boost: float = 2.0
    ess_target_train: Optional[float] = None

    seed: int = 42
    device: str = "cpu"

    cond_ortho_warmup_epochs: int = 15

    n_samples_hint: Optional[int] = None
    adaptive: bool = True

    def apply_theorem45_defaults(self) -> "IVAPCIV33TheoryConfig":
        n = self.n_samples_hint
        if (not self.adaptive) or (n is None) or (n <= 0):
            return self
        base = max(6, int(2 * np.log1p(n)))
        self.latent_x_dim = max(2, base // 3)
        self.latent_w_dim = max(2, base // 3)
        self.latent_z_dim = max(2, base // 6)
        self.latent_n_dim = max(2, base // 3)

        self.lambda_ortho = 0.005 * np.sqrt(np.log1p(n))
        self.lambda_consistency = 0.04 * np.sqrt(np.log1p(n))

        decay = 1.0 / np.sqrt(np.log1p(n))
        scale_decay = min(1.0, 1.5 * decay)
        self.gamma_adv_w = 0.15 * scale_decay
        self.gamma_adv_z = 0.12 * scale_decay
        self.gamma_adv_n = 0.10 * scale_decay

        self.gamma_padic = 0.001 * np.sqrt(np.log1p(n))
        self.min_std = max(1e-4, 1e-2 / np.sqrt(np.log1p(n)))
        self.low_var_min_std = min(self.low_var_min_std, self.min_std * 0.1)

        self.epochs_pretrain = int(min(80, 30 + 10 * np.log1p(n / 500)))
        self.epochs_main = int(min(220, 100 + 15 * np.log1p(n / 500)))
        # ---------- Overlap / ESS targets (Theorem 5 bias–variance trade-off) ----------
        # Keep training ESS targets conservative on small n to avoid aggressive clipping.
        if getattr(self, "ess_target_train", None) is None:
            r = float(np.log1p(n) / max(1e-6, np.log1p(2000.0)))
            self.ess_target_train = float(np.clip(0.25 + 0.18 * r, 0.25, 0.55))
        self.ess_target = float(np.clip(self.ess_target_train + 0.05, 0.30, 0.60))

        # Allow adaptive clipping to select the minimum threshold that reaches ESS goals.
        self.clip_prop = float(min(self.clip_prop, 0.01))
        self.clip_prop_adaptive_max = float(max(self.clip_prop_adaptive_max, 0.05))

        # Adversary steps: small datasets benefit from a couple of extra discriminator steps.
        if n <= 800:
            self.adv_steps = int(max(self.adv_steps_min, min(self.adv_steps_max, max(2, self.adv_steps))))
        else:
            self.adv_steps = int(max(self.adv_steps_min, min(self.adv_steps_max, self.adv_steps)))
        return self


def adaptive_regularization_schedule(
    epoch: int,
    total_epochs: int,
    diagnostics: dict,
    config: IVAPCIV33TheoryConfig,
) -> tuple[float, float, float]:
    """Dynamic adjustment of adversarial and HSIC weights.

    Balances Theorem 2 (information preservation) and Theorem 3 (exclusion/independence)
    with a gentle phase schedule:
    - Early phase: looser constraints for expressiveness
    - Mid phase: nominal weights
    - Late phase: slightly stronger constraints to tighten identifiability
    Diagnostic cues allow reacting to W→A leakage and Z-exclusion leakage.
    Returns (gamma_adv_w, gamma_adv_z, lambda_hsic).
    """

    base_w = config.gamma_adv_w
    base_z = config.gamma_adv_z
    base_hsic = config.lambda_hsic

    if epoch < total_epochs * 0.3:
        phase_scale = 0.7
    elif epoch < total_epochs * 0.7:
        phase_scale = 1.0
    else:
        phase_scale = 1.2

    info_loss = diagnostics.get("info_loss_proxy", 0.015)
    if info_loss > 0.018:
        info_scale = 0.8
    elif info_loss < 0.014:
        info_scale = 1.1
    else:
        info_scale = 1.0

    w_auc = diagnostics.get("rep_auc_w_to_a", 0.5)
    w_dev = abs(w_auc - 0.5)
    if w_dev > 0.18:
        w_boost = 1.5
    elif w_dev > 0.12:
        w_boost = 1.2
    elif w_dev < 0.08:
        w_boost = 0.9
    else:
        w_boost = 1.0

    z_leak = diagnostics.get("rep_exclusion_leakage_r2", 0.15)
    if z_leak > 0.20:
        z_boost = 1.5
    elif z_leak > 0.15:
        z_boost = 1.2
    elif z_leak < 0.10:
        z_boost = 0.9
    else:
        z_boost = 1.0

    gamma_w = base_w * phase_scale * info_scale * w_boost
    gamma_z = base_z * phase_scale * info_scale * z_boost
    lambda_h = base_hsic * phase_scale * info_scale

    gamma_w = float(np.clip(gamma_w, 0.05, 0.4))
    gamma_z = float(np.clip(gamma_z, 0.05, 0.3))
    lambda_h = float(np.clip(lambda_h, 0.005, 0.05))

    return gamma_w, gamma_z, lambda_h


class SmartAdversarialScheduler:
    """Warmup + cosine ramp + feedback scheduler for adversarial strengths."""

    def __init__(
        self,
        base_gamma_w: float,
        base_gamma_z: float,
        warmup_epochs: int = 10,
        ramp_epochs: int = 30,
        target_w_auc: float = 0.5,
        target_z_r2: float = 0.1,
    ) -> None:
        self.base_gamma_w = base_gamma_w
        self.base_gamma_z = base_gamma_z
        self.warmup = warmup_epochs
        self.ramp = ramp_epochs
        self.target_w_auc = target_w_auc
        self.target_z_r2 = target_z_r2

    def step(self, epoch: int, total_epochs: int, diagnostics: dict) -> tuple[float, float]:
        import numpy as _np

        if epoch < self.warmup:
            return 0.0, 0.0

        if epoch < self.warmup + self.ramp:
            prog = (epoch - self.warmup) / max(1, self.ramp)
            ramp_factor = 0.5 * (1 - _np.cos(_np.pi * prog))
        else:
            ramp_factor = 1.0

        base_w = self.base_gamma_w * ramp_factor
        base_z = self.base_gamma_z * ramp_factor

        w_auc = float(diagnostics.get("rep_auc_w_to_a", 0.5))
        z_r2 = float(diagnostics.get("rep_exclusion_leakage_r2", 0.0))

        w_dev = abs(w_auc - self.target_w_auc)
        if w_dev > 0.15:
            w_mult = 1.3
        elif w_dev > 0.08:
            w_mult = 1.1
        elif w_dev < 0.05:
            w_mult = 0.9
        else:
            w_mult = 1.0

        if z_r2 > 0.18:
            z_mult = 1.3
        elif z_r2 > 0.12:
            z_mult = 1.1
        elif z_r2 < 0.08:
            z_mult = 0.9
        else:
            z_mult = 1.0

        gamma_w = float(_np.clip(base_w * w_mult, 0.0, 0.4))
        gamma_z = float(_np.clip(base_z * z_mult, 0.0, 0.3))
        return gamma_w, gamma_z


# -------------------------
# Estimator (no RADR)
# -------------------------

class IVAPCIv33TheoryHierEstimator(BaseCausalEstimator):
    """Hierarchical encoder with theorem-1-5 components (DR stage)."""

    def __init__(self, config: Optional[IVAPCIV33TheoryConfig] = None):
        self.config = config or IVAPCIV33TheoryConfig()
        if self.config.n_samples_hint is not None:
            self.config.apply_theorem45_defaults()
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._is_fit = False
        self._protected_mask: Optional[np.ndarray] = None
        self._weak_iv_flag: bool = False
        self.training_diagnostics: Dict[str, float] = {}
        self.info_monitor = InformationLossMonitor()

    def _split_raw_blocks(self, V_all: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        cfg = self.config
        total = cfg.x_dim + cfg.w_dim + cfg.z_dim
        if total <= 0 or total > V_all.shape[1]:
            return None
        x = V_all[:, : cfg.x_dim]
        w = V_all[:, cfg.x_dim : cfg.x_dim + cfg.w_dim]
        z = V_all[:, cfg.x_dim + cfg.w_dim : cfg.x_dim + cfg.w_dim + cfg.z_dim]
        return x, w, z

    def _identifiability_checks(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        """Lightweight IV relevance/exclusion proxies stored in diagnostics."""
        blocks = self._split_raw_blocks(V_all)
        if blocks is None:
            return
        X_raw, W_raw, Z_raw = blocks
        diag: Dict[str, float] = {}

        def _safe_corr(u: np.ndarray, v: np.ndarray) -> float:
            if u.size == 0 or v.size == 0 or np.std(u) == 0 or np.std(v) == 0:
                return float("nan")
            return float(np.corrcoef(u, v)[0, 1])

        # IV relevance: average abs corr between Z cols and A
        corrs = [_safe_corr(Z_raw[:, j], A) for j in range(Z_raw.shape[1])]
        diag["iv_relevance_abs_corr"] = float(np.nanmean(np.abs(corrs))) if corrs else float("nan")

        # First-stage F-statistic proxy: A ~ [X,W,Z] vs reduced A ~ [X,W]
        try:
            XW = np.column_stack([np.ones(len(A)), X_raw, W_raw])
            XWZ = np.column_stack([np.ones(len(A)), X_raw, W_raw, Z_raw])
            beta_r, *_ = np.linalg.lstsq(XW, A, rcond=None)
            beta_f, *_ = np.linalg.lstsq(XWZ, A, rcond=None)
            resid_r = A - XW @ beta_r
            resid_f = A - XWZ @ beta_f
            rss_r = float(np.sum(resid_r**2))
            rss_f = float(np.sum(resid_f**2))
            k = Z_raw.shape[1]
            n = len(A)
            f_num = (rss_r - rss_f) / max(k, 1)
            f_den = rss_f / max(n - XWZ.shape[1], 1)
            f_stat = f_num / (f_den + 1e-12)
            diag["iv_first_stage_f"] = float(f_stat)
            if f_stat < 10.0:
                diag["weak_iv_warning"] = "First-stage F < 10: weak IV detected; using conservative DR."
                self._weak_iv_flag = True
        except Exception:
            diag["iv_first_stage_f"] = float("nan")

        diag["weak_iv_flag"] = bool(getattr(self, "_weak_iv_flag", False))

        # Exclusion proxy: corr between Z and residual of Y ~ [A, X, W]
        if X_raw.size and W_raw.size:
            cov = np.column_stack([np.ones(len(A)), A, X_raw, W_raw])
            try:
                coef, *_ = np.linalg.lstsq(cov, Y, rcond=None)
                resid = Y - cov @ coef
                ex_corrs = [_safe_corr(Z_raw[:, j], resid) for j in range(Z_raw.shape[1])]
                diag["iv_exclusion_abs_corr_resid"] = float(np.nanmean(np.abs(ex_corrs)))
            except np.linalg.LinAlgError:
                diag["iv_exclusion_abs_corr_resid"] = float("nan")
        self.training_diagnostics.update(diag)

    def _post_fit_quality_diagnostics(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        """Quick holdout diagnostics for learned blocks (AUC/R2).

        These are *diagnostics*, not formal identification tests:
        - AUC(tz -> A): should be reasonably high if tz captures treatment-relevant signal.
        - AUC(tw -> A): should be near 0.5 if W is disentangled from treatment.
        - R2([A, tx, tw] -> Y): should be higher than R2([A, tz] -> Y).
        - R2([A, tz] -> Y): proxy for exclusion leakage (lower is better).
        """
        if not hasattr(self, "training_diagnostics"):
            self.training_diagnostics = {}

        try:
            V_all = np.asarray(V_all, dtype=np.float32)
            A = np.asarray(A).reshape(-1)
            Y = np.asarray(Y).reshape(-1)
            n = len(A)
            if n < 20:
                return

            V_std = _apply_standardize(V_all, self._v_mean, self._v_std)
            with torch.no_grad():
                V_t = torch.from_numpy(V_std).to(self.device)
                tx, tw, tz, _tn = self._encode_blocks(V_t)
                txn = tx.detach().cpu().numpy()
                twn = tw.detach().cpu().numpy()
                tzn = tz.detach().cpu().numpy()

            a_vals = np.unique(A)
            is_binary01 = bool(a_vals.size == 2 and np.all(np.isin(a_vals, [0, 1])))
            A01 = A.astype(int) if is_binary01 else None
            strat = A01 if (A01 is not None and np.min(np.bincount(A01)) >= 2) else None
            idx = np.arange(n)
            idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=0, stratify=strat)

            def _auc(feat: np.ndarray) -> float:
                if A01 is None or feat.size == 0:
                    return float("nan")
                lr = LogisticRegression(max_iter=1000)
                lr.fit(feat[idx_tr], A01[idx_tr])
                p = lr.predict_proba(feat[idx_te])[:, 1]
                return float(roc_auc_score(A01[idx_te], p))

            def _r2(feat: np.ndarray) -> float:
                if feat.size == 0 or np.var(Y[idx_te]) == 0:
                    return float("nan")
                reg = Ridge(alpha=1.0)
                reg.fit(feat[idx_tr], Y[idx_tr])
                pred = reg.predict(feat[idx_te])
                return float(r2_score(Y[idx_te], pred))

            auc_z_a = _auc(tzn) if tzn.ndim == 2 else _auc(tzn.reshape(-1, 1))
            auc_w_a = _auc(twn) if twn.ndim == 2 else _auc(twn.reshape(-1, 1))

            feat_xw_a = np.column_stack([A, txn, twn])
            feat_z_a = np.column_stack([A, tzn])

            r2_xw_a_y = _r2(feat_xw_a)
            r2_z_a_y = _r2(feat_z_a)

            self.training_diagnostics.update(
                {
                    "rep_auc_z_to_a": auc_z_a,
                    "rep_auc_w_to_a": auc_w_a,
                    "rep_r2_xw_a_to_y": r2_xw_a_y,
                    "rep_r2_z_a_to_y": r2_z_a_y,
                    "rep_exclusion_leakage_r2": float(r2_z_a_y) if np.isfinite(r2_z_a_y) else float("nan"),
                    "rep_exclusion_leakage_gap": float(r2_z_a_y - r2_xw_a_y)
                    if (np.isfinite(r2_z_a_y) and np.isfinite(r2_xw_a_y))
                    else float("nan"),
                }
            )
        except Exception:
            # Diagnostics should never crash fitting.
            return

    @staticmethod
    def _toggle_requires_grad(modules: Iterable[nn.Module], flag: bool) -> None:
        for mod in modules:
            for p in mod.parameters():
                p.requires_grad = flag

    # --- model wiring ---
    def _build(self, d_all: int) -> None:
        cfg = self.config
        if not (cfg.x_dim and cfg.w_dim and cfg.z_dim):
            raise ValueError("IVAPCI v3.3 requires x_dim, w_dim, z_dim to split X|W|Z blocks")
        if cfg.x_dim + cfg.w_dim + cfg.z_dim != d_all:
            raise ValueError("x_dim + w_dim + z_dim must equal input dimension")

        self._block_slices = (
            slice(0, cfg.x_dim),
            slice(cfg.x_dim, cfg.x_dim + cfg.w_dim),
            slice(cfg.x_dim + cfg.w_dim, cfg.x_dim + cfg.w_dim + cfg.z_dim),
        )

        self.enc_x = _GroupEncoder(cfg.x_dim, cfg.enc_x_hidden, cfg.latent_x_dim).to(self.device)
        self.enc_w = _GroupEncoder(cfg.w_dim, cfg.enc_w_hidden, cfg.latent_w_dim).to(self.device)
        self.enc_z = _GroupEncoder(cfg.z_dim, cfg.enc_z_hidden, cfg.latent_z_dim, dropout=cfg.dropout_z).to(
            self.device
        )
        self.enc_n = _GroupEncoder(d_all, cfg.enc_n_hidden, cfg.latent_n_dim).to(self.device)

        total_lat = cfg.latent_x_dim + cfg.latent_w_dim + cfg.latent_z_dim + cfg.latent_n_dim
        self.decoder = _Decoder(total_lat, cfg.dec_hidden, d_all).to(self.device)

        self.a_head = _AClassifier(cfg.latent_x_dim + cfg.latent_z_dim, cfg.a_hidden).to(self.device)
        self.y_head = _YRegressor(cfg.latent_x_dim + cfg.latent_w_dim, cfg.y_hidden).to(self.device)

        self.a_from_z = _AClassifier(cfg.latent_z_dim, (32,)).to(self.device)
        self.y_from_w = _YRegressor(cfg.latent_w_dim, (64, 32)).to(self.device)

        self.adv_w = _AClassifier(cfg.latent_w_dim, cfg.adv_a_hidden).to(self.device)
        self.adv_n = _AClassifier(cfg.latent_n_dim, cfg.adv_a_hidden).to(self.device)
        self.adv_z = _YAdversary(cfg.latent_z_dim, cfg.adv_y_hidden).to(self.device)
        self.adv_w_cond = TrueConditionalAdversary(
            cfg.latent_w_dim, cfg.latent_x_dim, cfg.adv_a_hidden, loss="bce"
        ).to(self.device)
        self.adv_z_cond = TrueConditionalAdversary(
            cfg.latent_z_dim, cfg.latent_x_dim + 1, cfg.adv_y_hidden, loss="mse"
        ).to(self.device)
        self.adv_n_cond = TrueConditionalAdversary(
            cfg.latent_n_dim, cfg.latent_x_dim + cfg.latent_w_dim + cfg.latent_z_dim, cfg.adv_a_hidden, loss="bce"
        ).to(self.device)

        main_params = (
            list(self.enc_x.parameters())
            + list(self.enc_w.parameters())
            + list(self.enc_z.parameters())
            + list(self.enc_n.parameters())
            + list(self.decoder.parameters())
            + list(self.a_head.parameters())
            + list(self.y_head.parameters())
            + list(self.a_from_z.parameters())
            + list(self.y_from_w.parameters())
        )
        self.main_opt = torch.optim.Adam(main_params, lr=cfg.lr_main)
        self.main_sched = ReduceLROnPlateau(self.main_opt, mode="min", factor=0.5, patience=10)

        adv_params = (
            list(self.adv_w.parameters())
            + list(self.adv_z.parameters())
            + list(self.adv_n.parameters())
            + list(self.adv_w_cond.parameters())
            + list(self.adv_z_cond.parameters())
            + list(self.adv_n_cond.parameters())
        )
        self.adv_opt = torch.optim.Adam(adv_params, lr=cfg.lr_adv)

    def _split_blocks_tensor(
        self, V_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._block_slices is not None, "Block slices must be initialized in _build"
        return (
            V_t[:, self._block_slices[0]],
            V_t[:, self._block_slices[1]],
            V_t[:, self._block_slices[2]],
        )

    def _encode_blocks(
        self, V_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_dtype = next(self.enc_x.parameters()).dtype
        if V_t.dtype != target_dtype:
            V_t = V_t.to(dtype=target_dtype)
        x_part, w_part, z_part = self._split_blocks_tensor(V_t)
        tx = self.enc_x(x_part)
        tw = self.enc_w(w_part)
        tz = self.enc_z(z_part)
        tn = self.enc_n(V_t)
        return tx, tw, tz, tn

    # --- training ---
    def fit(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        V_all = np.asarray(V_all, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32).reshape(-1)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)
        n, d_all = V_all.shape
        cfg = self.config

        # reset weak-IV flag per fit
        self._weak_iv_flag = False

        if cfg.n_samples_hint is None:
            cfg.n_samples_hint = int(n)
            if cfg.adaptive:
                cfg.apply_theorem45_defaults()

        # Pre-training identifiability checks (lightweight proxies)
        self._identifiability_checks(V_all, A, Y)

        # stratify only when each class has enough mass for the holdout
        uniq, counts = np.unique(A, return_counts=True)
        stratify_arr = None
        if uniq.size > 1:
            test_size = max(1, int(round(cfg.val_frac * n)))
            if test_size >= uniq.size and counts.min() >= 2:
                stratify_arr = A
        tr_idx, va_idx = train_test_split(
            np.arange(n),
            test_size=cfg.val_frac,
            random_state=cfg.seed,
            stratify=stratify_arr,
        )
        V_tr, V_va = V_all[tr_idx], V_all[va_idx]
        A_tr, A_va = A[tr_idx], A[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]

        V_tr_std, self._v_mean, self._v_std, protected = _info_aware_standardize(
            V_tr,
            min_std=cfg.min_std,
            low_var_min_std=cfg.low_var_min_std,
            clip_value=cfg.std_clip,
        )
        self._protected_mask = protected
        V_va_std = _apply_standardize(V_va, self._v_mean, self._v_std)
        if cfg.std_clip is not None:
            V_tr_std = np.clip(V_tr_std, -cfg.std_clip, cfg.std_clip)
            V_va_std = np.clip(V_va_std, -cfg.std_clip, cfg.std_clip)

        Y_tr_std, self._y_mean, self._y_std, _ = _info_aware_standardize(
            Y_tr.reshape(-1, 1), min_std=1e-6, low_var_min_std=1e-6, clip_value=cfg.std_clip
        )
        Y_tr_std = Y_tr_std.squeeze(1)
        Y_va_std = _apply_standardize(Y_va.reshape(-1, 1), self._y_mean, self._y_std).squeeze(1)

        self._build(d_all)
        target_dtype = next(self.enc_x.parameters()).dtype

        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        tr_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(V_tr_std), torch.from_numpy(A_tr), torch.from_numpy(Y_tr_std)
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        va_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(V_va_std), torch.from_numpy(A_va), torch.from_numpy(Y_va_std)
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        self._adv_scheduler = SmartAdversarialScheduler(
            base_gamma_w=cfg.gamma_adv_w,
            base_gamma_z=cfg.gamma_adv_z,
            warmup_epochs=cfg.adv_warmup_epochs,
            ramp_epochs=cfg.adv_ramp_epochs,
        )

        # Stage 0: reconstruction
        for _ in range(cfg.epochs_pretrain):
            self.enc_x.train(); self.enc_w.train(); self.enc_z.train(); self.enc_n.train(); self.decoder.train()
            for vb, _, _ in tr_loader:
                vb = vb.to(self.device)
                tx, tw, tz, tn = self._encode_blocks(vb)
                recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
                loss = mse(recon, vb)
                self.main_opt.zero_grad(); loss.backward(); self.main_opt.step()

        best_val = float("inf")
        patience = 0
        best_state = None

        # Stage 1: full training
        for epoch in range(cfg.epochs_main):
            self.enc_x.train(); self.enc_w.train(); self.enc_z.train(); self.enc_n.train()
            self.decoder.train(); self.a_head.train(); self.y_head.train()
            self.a_from_z.train(); self.y_from_w.train()
            self.adv_w.train(); self.adv_z.train(); self.adv_n.train()
            self.adv_w_cond.train(); self.adv_z_cond.train(); self.adv_n_cond.train()

            _, _, lambda_hsic_ep = adaptive_regularization_schedule(
                epoch, cfg.epochs_main, getattr(self, "training_diagnostics", {}), cfg
            )
            gamma_w_use, gamma_z_use = self._adv_scheduler.step(
                epoch, cfg.epochs_main, getattr(self, "training_diagnostics", {})
            )
            w_factor = gamma_w_use / max(cfg.gamma_adv_w, 1e-8)
            z_factor = gamma_z_use / max(cfg.gamma_adv_z, 1e-8)
            cond_factor = 0.0
            if epoch >= cfg.cond_adv_warmup_epochs:
                cond_factor = float(
                    min(1.0, (epoch - cfg.cond_adv_warmup_epochs + 1) / max(1, cfg.cond_adv_ramp_epochs))
                )
            gamma_n_use = cfg.gamma_adv_n * w_factor
            gamma_w_cond = cfg.gamma_adv_w_cond * w_factor * cond_factor
            gamma_z_cond = cfg.gamma_adv_z_cond * z_factor * cond_factor
            gamma_n_cond = cfg.gamma_adv_n_cond * w_factor * cond_factor

            adv_steps_ep = cfg.adv_steps
            if cfg.adv_steps_dynamic:
                diag_now = getattr(self, "training_diagnostics", {}) or {}
                w_auc = float(diag_now.get("rep_auc_w_to_a", 0.5))
                z_r2 = float(diag_now.get("rep_exclusion_leakage_r2", 0.0))
                if w_auc > 0.62 or z_r2 > 0.18:
                    adv_steps_ep += 1
                if w_auc > 0.68 or z_r2 > 0.25:
                    adv_steps_ep += 1
                adv_steps_ep = int(np.clip(adv_steps_ep, cfg.adv_steps_min, cfg.adv_steps_max))

            epoch_loss = 0.0
            epoch_batches = 0
            for vb, ab, yb in tr_loader:
                vb = vb.to(self.device, dtype=target_dtype)
                ab = ab.to(self.device, dtype=target_dtype)
                yb = yb.to(self.device, dtype=target_dtype)

                # adversaries update (multi-step, conditional)
                if adv_steps_ep > 0 and (gamma_w_use > 0 or gamma_z_use > 0 or gamma_n_use > 0):
                    for _ in range(adv_steps_ep):
                        with torch.no_grad():
                            tx_d, tw_d, tz_d, tn_d = self._encode_blocks(vb)
                        adv_loss = (
                            bce(self.adv_w(tw_d), ab)
                            + bce(self.adv_n(tn_d), ab)
                            + mse(self.adv_z(tz_d), yb)
                        )
                        if cond_factor > 0:
                            adv_loss = adv_loss + self.adv_w_cond(tw_d, tx_d, ab)
                            adv_loss = adv_loss + self.adv_z_cond(
                                tz_d, torch.cat([tx_d, ab.unsqueeze(1)], dim=1), yb
                            )
                            adv_loss = adv_loss + self.adv_n_cond(
                                tn_d, torch.cat([tx_d, tw_d, tz_d], dim=1), ab
                            )
                        self.adv_opt.zero_grad(); adv_loss.backward(); self.adv_opt.step()

                # main update
                self._toggle_requires_grad(
                    [self.adv_w, self.adv_n, self.adv_z, self.adv_w_cond, self.adv_z_cond, self.adv_n_cond],
                    False,
                )
                tx, tw, tz, tn = self._encode_blocks(vb)
                recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
                logits_a = self.a_head(torch.cat([tx, tz], dim=1))
                y_pred = self.y_head(torch.cat([tx, tw], dim=1), ab)

                cons_a = bce(self.a_from_z(tz), ab)
                cons_y = mse(self.y_from_w(tw, ab), yb)
                consistency = cons_a + cons_y

                ortho = _offdiag_corr_penalty([tx, tw, tz, tn])
                warmup = cfg.cond_ortho_warmup_epochs
                cond_weight = 0.0
                if cfg.lambda_cond_ortho > 0 and epoch >= warmup:
                    ramp = min(1.0, (epoch - warmup + 1) / max(1, warmup))
                    cond_weight = float(cfg.lambda_cond_ortho * ramp)
                cond_ortho = torch.zeros((), device=self.device)
                if cond_weight > 0:
                    # W ⟂ Z | X and noise ⟂ (X,W,Z)
                    cond_ortho_wz = _conditional_orthogonal_penalty(
                        [tw, tz], tx, ridge=cfg.ridge_alpha
                    )
                    cond_ortho_n = _conditional_orthogonal_penalty(
                        [tn], torch.cat([tx, tw, tz], dim=1), ridge=cfg.ridge_alpha
                    )
                    cond_ortho = cond_ortho_wz + cond_ortho_n

                tx_det = tx.detach()
                tw_det = tw.detach()
                tz_det = tz.detach()
                adv_w_logits = self.adv_w(tw)
                adv_n_logits = self.adv_n(tn)
                adv_z_pred = self.adv_z(tz)
                adv_w_cond_loss = None
                adv_z_cond_loss = None
                adv_n_cond_loss = None
                if cond_factor > 0:
                    adv_w_cond_loss = self.adv_w_cond(tw, tx_det, ab)
                    adv_z_cond_loss = self.adv_z_cond(
                        tz, torch.cat([tx_det, ab.unsqueeze(1)], dim=1), yb
                    )
                    adv_n_cond_loss = self.adv_n_cond(
                        tn, torch.cat([tx_det, tw_det, tz_det], dim=1), ab
                    )

                hsic_pen = torch.zeros((), device=self.device)
                if lambda_hsic_ep > 0:
                    # subsample for stability
                    if tx.shape[0] > cfg.hsic_max_samples:
                        idx = torch.randperm(tx.shape[0], device=self.device)[: cfg.hsic_max_samples]
                        tx_h, tw_h, tz_h, tn_h = tx[idx], tw[idx], tz[idx], tn[idx]
                    else:
                        tx_h, tw_h, tz_h, tn_h = tx, tw, tz, tn
                    hsic_pen = (
                        _rbf_hsic(tx_h, tw_h)
                        + _rbf_hsic(tw_h, tz_h)
                        + _rbf_hsic(tx_h, tn_h)
                        + _rbf_hsic(tw_h, tn_h)
                        + _rbf_hsic(tz_h, tn_h)
                    )

                overlap_pen = torch.zeros((), device=self.device)
                if cfg.lambda_overlap > 0 and epoch >= cfg.overlap_warmup_epochs:
                    prob_a = torch.sigmoid(logits_a)
                    m = cfg.overlap_margin
                    overlap_pen = torch.mean(torch.relu(m - prob_a) ** 2 + torch.relu(m - (1 - prob_a)) ** 2)

                loss_main = (
                    cfg.lambda_recon * mse(recon, vb)
                    + cfg.lambda_a * bce(logits_a, ab)
                    + cfg.lambda_y * mse(y_pred, yb)
                    + cfg.lambda_consistency * consistency
                    + cfg.lambda_ortho * ortho
                    + cond_weight * cond_ortho
                    + cfg.gamma_padic * _padic_ultrametric_loss(torch.cat([tx, tz], dim=1))
                    + lambda_hsic_ep * hsic_pen
                    + cfg.lambda_overlap * overlap_pen
                    - gamma_w_use * bce(adv_w_logits, ab)
                    - gamma_n_use * bce(adv_n_logits, ab)
                    - gamma_z_use * mse(adv_z_pred, yb)
                )
                if cond_factor > 0:
                    if adv_w_cond_loss is not None:
                        loss_main = loss_main - gamma_w_cond * adv_w_cond_loss
                    if adv_z_cond_loss is not None:
                        loss_main = loss_main - gamma_z_cond * adv_z_cond_loss
                    if adv_n_cond_loss is not None:
                        loss_main = loss_main - gamma_n_cond * adv_n_cond_loss

                self.main_opt.zero_grad(); loss_main.backward(); self.main_opt.step()
                self._toggle_requires_grad(
                    [self.adv_w, self.adv_n, self.adv_z, self.adv_w_cond, self.adv_z_cond, self.adv_n_cond],
                    True,
                )
                epoch_loss += float(loss_main.item())
                epoch_batches += 1

            # validation
            self.enc_x.eval(); self.enc_w.eval(); self.enc_z.eval(); self.enc_n.eval()
            self.decoder.eval(); self.a_head.eval(); self.y_head.eval()
            vals = []
            with torch.no_grad():
                for vb, ab, yb in va_loader:
                    vb = vb.to(self.device, dtype=target_dtype)
                    ab = ab.to(self.device, dtype=target_dtype)
                    yb = yb.to(self.device, dtype=target_dtype)
                    tx, tw, tz, tn = self._encode_blocks(vb)
                    recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
                    logits_a = self.a_head(torch.cat([tx, tz], dim=1))
                    y_pred = self.y_head(torch.cat([tx, tw], dim=1), ab)
                    loss_val = (
                        cfg.lambda_recon * mse(recon, vb)
                        + cfg.lambda_a * bce(logits_a, ab)
                        + cfg.lambda_y * mse(y_pred, yb)
                    )
                    vals.append(float(loss_val.item()))
            mean_val = float(np.mean(vals)) if vals else float("inf")
            self.main_sched.step(mean_val)

            if epoch_batches > 0 and epoch % 10 == 0:
                try:
                    # quick validation-based diagnostics for monitoring
                    self._post_fit_quality_diagnostics(V_all=V_va, A=A_va, Y=Y_va)
                except Exception:
                    pass
                enhanced_training_monitor(
                    self,
                    epoch,
                    {
                        "main": epoch_loss / max(1, epoch_batches),
                        "val": mean_val,
                    },
                )

            if mean_val + cfg.early_stopping_min_delta < best_val:
                best_val = mean_val
                patience = 0
                best_state = {k: v.state_dict() for k, v in {
                    "enc_x": self.enc_x, "enc_w": self.enc_w, "enc_z": self.enc_z, "enc_n": self.enc_n,
                    "decoder": self.decoder, "a_head": self.a_head, "y_head": self.y_head,
                    "a_from_z": self.a_from_z, "y_from_w": self.y_from_w,
                    "adv_w": self.adv_w, "adv_z": self.adv_z, "adv_n": self.adv_n,
                    "adv_w_cond": self.adv_w_cond, "adv_z_cond": self.adv_z_cond, "adv_n_cond": self.adv_n_cond,
                }.items()}
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break

        if best_state is not None:
            self.enc_x.load_state_dict(best_state["enc_x"])
            self.enc_w.load_state_dict(best_state["enc_w"])
            self.enc_z.load_state_dict(best_state["enc_z"])
            self.enc_n.load_state_dict(best_state["enc_n"])
            self.decoder.load_state_dict(best_state["decoder"])
            self.a_head.load_state_dict(best_state["a_head"])
            self.y_head.load_state_dict(best_state["y_head"])
            self.a_from_z.load_state_dict(best_state["a_from_z"])
            self.y_from_w.load_state_dict(best_state["y_from_w"])
            self.adv_w.load_state_dict(best_state["adv_w"])
            self.adv_z.load_state_dict(best_state["adv_z"])
            self.adv_n.load_state_dict(best_state["adv_n"])
            self.adv_w_cond.load_state_dict(best_state["adv_w_cond"])
            self.adv_z_cond.load_state_dict(best_state["adv_z_cond"])
            self.adv_n_cond.load_state_dict(best_state["adv_n_cond"])

        self._is_fit = True

        # Post-fit quality diagnostics on learned representations (quick holdout)
        self._post_fit_quality_diagnostics(V_all, A, Y)

        # Diagnostics (keep any diagnostics collected during fit)
        if not hasattr(self, "training_diagnostics"):
            self.training_diagnostics = {}
        self.training_diagnostics.update(self.info_monitor.estimate(self, V_all))
        self.training_diagnostics["protected_features"] = (
            int(np.sum(self._protected_mask)) if self._protected_mask is not None else 0
        )
        self.training_diagnostics["min_std"] = float(cfg.min_std)

        # Post-fit representation diagnostics (predictability/leakage checks)
        try:
            self._post_fit_quality_diagnostics(V_all=V_all, A=A, Y=Y)
        except Exception as e:
            self.training_diagnostics["post_fit_diag_error"] = f"{type(e).__name__}: {e}"

        # Adversary diagnostics (use raw-scale targets for comparability)
        V_std = _apply_standardize(V_all.astype(np.float32), self._v_mean, self._v_std)
        V_t = torch.from_numpy(V_std).to(self.device)

        self.enc_x.eval(); self.enc_w.eval(); self.enc_z.eval(); self.enc_n.eval()
        self.adv_w.eval(); self.adv_n.eval(); self.adv_z.eval()
        self.adv_w_cond.eval(); self.adv_z_cond.eval(); self.adv_n_cond.eval()
        with torch.no_grad():
            tx, tw, tz, tn = self._encode_blocks(V_t)
            adv_w_probs = torch.sigmoid(self.adv_w(tw)).cpu().numpy()
            adv_n_probs = torch.sigmoid(self.adv_n(tn)).cpu().numpy()
            adv_z_pred = self.adv_z(tz).cpu().numpy()

        adv_w_acc = float(((adv_w_probs > 0.5) == (A == 1)).mean()) if adv_w_probs.size else np.nan
        adv_n_acc = float(((adv_n_probs > 0.5) == (A == 1)).mean()) if adv_n_probs.size else np.nan

        # adv_z is trained on standardized Y; convert back to raw scale for diagnostics
        if hasattr(self, "_y_std") and hasattr(self, "_y_mean"):
            y_scale = float(self._y_std.squeeze())
            y_mean = float(self._y_mean.squeeze())
            adv_z_pred_raw = adv_z_pred * y_scale + y_mean
        else:
            adv_z_pred_raw = adv_z_pred
        adv_z_r2 = float(r2_score(Y, adv_z_pred_raw)) if np.var(Y) > 0 else np.nan

        self.training_diagnostics.update(
            {
                "adv_w_acc": adv_w_acc,
                "adv_n_acc": adv_n_acc,
                "adv_z_r2": adv_z_r2,
            }
        )

        # Theorem-aligned diagnostics (adds theorem1/theorem2/theorem3_* keys)
        try:
            tcfg = TheoremDiagnosticsConfig(
                max_n_for_mi=min(5000, int(V_all.shape[0])),
                random_state=int(getattr(cfg, "seed", 0) or 0),
            )
            tdiag = TheoremComplianceDiagnostics(self, config=tcfg)
            t_res = tdiag.run_all_diagnostics(
                X_all=V_all,
                A=A,
                Y=Y,
                n_recon_features=min(5, int(V_all.shape[1])),
            )
            self.training_diagnostics.update(t_res)
        except Exception as e:  # pragma: no cover - diagnostics best-effort
            self.training_diagnostics["theorem_diag_error"] = str(e)[:200]

    # --- latent + DR ---
    def get_latent(self, V_all: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before get_latent.")
        V_all = np.asarray(V_all, dtype=np.float32)
        V_std = _apply_standardize(V_all, self._v_mean, self._v_std)
        V_t = torch.from_numpy(V_std).to(self.device)
        self.enc_x.eval(); self.enc_w.eval(); self.enc_z.eval(); self.enc_n.eval()
        with torch.no_grad():
            tx, tw, tz, tn = self._encode_blocks(V_t)
        if self.config.use_noise_in_latent:
            return torch.cat([tx, tw, tz, tn], dim=1).cpu().numpy()
        return torch.cat([tx, tw, tz], dim=1).cpu().numpy()

    def _dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        cfg = self.config
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        weak_iv = bool(getattr(self, "_weak_iv_flag", False))
        dr_mode = "weak_iv_conservative" if weak_iv else "default"
        stats = {k: [] for k in [
            "e_min", "e_max", "e_q01", "e_q05", "e_q95", "e_q99",
            "clip_used", "cap_used", "frac_e_clipped", "overlap_score", "ess_raw",
            "ess_overall_ratio", "ess_min", "clip_threshold",
            "ipw_abs_max_raw", "ipw_abs_max_postclip_precap", "ipw_abs_max_capped", "frac_ipw_capped",
        ]}

        def _quantiles(x: np.ndarray) -> Dict[str, float]:
            if x.size == 0:
                return {"min": np.nan, "max": np.nan, "q01": np.nan, "q05": np.nan, "q95": np.nan, "q99": np.nan}
            return {
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "q01": float(np.quantile(x, 0.01)),
                "q05": float(np.quantile(x, 0.05)),
                "q95": float(np.quantile(x, 0.95)),
                "q99": float(np.quantile(x, 0.99)),
            }

        kf = KFold(n_splits=cfg.n_splits_dr, shuffle=True, random_state=cfg.seed)
        psi = np.zeros_like(Y, dtype=float)
        for tr, te in kf.split(U):
            U_tr, U_te = U[tr], U[te]
            A_tr, A_te = A[tr], A[te]
            Y_tr, Y_te = Y[tr], Y[te]

            if cfg.standardize_nuisance:
                scaler_u = StandardScaler().fit(U_tr)
                U_tr_s = scaler_u.transform(U_tr)
                U_te_s = scaler_u.transform(U_te)
            else:
                U_tr_s, U_te_s = U_tr, U_te
            if np.unique(A_tr).size < 2:
                e_raw = np.full_like(A_te, float(np.mean(A_tr)) if len(A_tr) else 0.5, dtype=float)
            else:
                prop = LogisticRegression(max_iter=2000, solver="lbfgs", C=cfg.propensity_logreg_C)
                prop.fit(U_tr_s, A_tr)
                e_raw = prop.predict_proba(U_te_s)[:, 1]
            if cfg.propensity_shrinkage > 0:
                prior = float(np.mean(A_tr)) if len(A_tr) else 0.5
                e_raw = (1 - cfg.propensity_shrinkage) * e_raw + cfg.propensity_shrinkage * prior
            min_clip = max(float(cfg.clip_prop), 0.05) if weak_iv else float(cfg.clip_prop)
            max_clip = max(min_clip, float(cfg.clip_prop_adaptive_max))
            clip_use, clip_stats = find_optimal_clip_threshold(
                e_raw=e_raw,
                A=A_te,
                ess_target=float(cfg.ess_target),
                min_clip=min_clip,
                max_clip=max_clip,
                n_search=20,
            )
            stats["ess_raw"].append(float(clip_stats.get("ess_overall_ratio", np.nan)))
            stats["ess_overall_ratio"].append(float(clip_stats.get("ess_overall_ratio", np.nan)))
            stats["ess_min"].append(float(clip_stats.get("ess_min", np.nan)))
            stats["clip_threshold"].append(float(clip_stats.get("clip_threshold", clip_use)))
            cap_use = float(cfg.ipw_cap) if cfg.ipw_cap is not None else np.nan
            if weak_iv and np.isfinite(cap_use):
                cap_use = min(cap_use, 10.0)
            e_hat = np.clip(e_raw, clip_use, 1 - clip_use)

            qs = _quantiles(e_raw)
            for k, v in qs.items():
                stats[f"e_{k}"].append(v)
            stats["frac_e_clipped"].append(float(np.mean((e_raw < clip_use) | (e_raw > 1 - clip_use))))
            stats["overlap_score"].append(float(np.mean((e_raw > clip_use) & (e_raw < 1 - clip_use))))

            # Arm-specific outcome models to accommodate heterogeneous effects
            if np.any(A_tr == 1):
                m1_model: Ridge | DummyRegressor = Ridge(alpha=cfg.ridge_alpha)
                m1_model.fit(U_tr_s[A_tr == 1], Y_tr[A_tr == 1])
            else:
                m1_model = DummyRegressor(strategy="constant", constant=float(np.mean(Y_tr)))
                m1_model.fit(np.zeros((1, U_tr.shape[1])), [0.0])

            if np.any(A_tr == 0):
                m0_model: Ridge | DummyRegressor = Ridge(alpha=cfg.ridge_alpha)
                m0_model.fit(U_tr_s[A_tr == 0], Y_tr[A_tr == 0])
            else:
                m0_model = DummyRegressor(strategy="constant", constant=float(np.mean(Y_tr)))
                m0_model.fit(np.zeros((1, U_tr.shape[1])), [0.0])

            m1 = m1_model.predict(U_te_s)
            m0 = m0_model.predict(U_te_s)
            m_hat = np.where(A_te == 1, m1, m0)

            eps_raw = 1e-6
            w_raw = (A_te - np.clip(e_raw, eps_raw, 1 - eps_raw)) / (
                np.clip(e_raw, eps_raw, 1 - eps_raw)
                * (1 - np.clip(e_raw, eps_raw, 1 - eps_raw))
            )
            stats["ipw_abs_max_raw"].append(float(np.max(np.abs(w_raw))) if w_raw.size else np.nan)

            w = (A_te - e_hat) / (e_hat * (1 - e_hat))
            cap_used = cap_use
            if cfg.ipw_cap_quantile is not None:
                cap_used = adaptive_ipw_cap_by_quantile(
                    weights=w,
                    quantile=float(cfg.ipw_cap_quantile),
                    min_cap=5.0,
                    max_cap=float(cfg.ipw_cap) if cfg.ipw_cap is not None else 50.0,
                )
            stats["ipw_abs_max_postclip_precap"].append(float(np.max(np.abs(w))) if w.size else np.nan)
            if np.isfinite(cap_used) and cap_used > 0:
                w = np.clip(w, -float(cap_used), float(cap_used))
                stats["ipw_abs_max_capped"].append(float(np.max(np.abs(w))) if w.size else np.nan)
                stats["frac_ipw_capped"].append(
                    float(np.mean(np.abs(w_raw) >= float(cap_used))) if w_raw.size else np.nan
                )
            else:
                stats["ipw_abs_max_capped"].append(np.nan)
                stats["frac_ipw_capped"].append(np.nan)

            stats["clip_used"].append(clip_use)
            stats["cap_used"].append(cap_used)

            psi[te] = m1 - m0 + w * (Y_te - m_hat)
        if stats and hasattr(self, "training_diagnostics"):
            agg = {k: float(np.nanmean(v)) for k, v in stats.items() if v}
            self.training_diagnostics.update({f"dr_{k}": v for k, v in agg.items()})
            self.training_diagnostics["dr_mode"] = dr_mode
        return float(np.mean(psi))

    def estimate_ate(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        U = self.get_latent(V_all)
        return self._dr_ate(U, A, Y)

    def get_training_diagnostics(self) -> Dict[str, float]:
        return dict(self.training_diagnostics)


class IVAPCIv33TheoryHierRADREstimator(IVAPCIv33TheoryHierEstimator):
    """Same encoder, plus RADR-style calibration with fold-specific nuisances."""

    # ---- helpers for RADR features ----
    def _destandardize_y(self, y_std: np.ndarray) -> np.ndarray:
        scale = float(self._y_std.squeeze()) if hasattr(self, "_y_std") else 1.0
        mean = float(self._y_mean.squeeze()) if hasattr(self, "_y_mean") else 0.0
        return y_std * scale + mean

    def _head_features(
        self, U: np.ndarray, A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propensity/outcome head predictions for RADR nuisance models."""

        tx, tw, tz, tn = self._split_latent(U)
        A = np.asarray(A, dtype=np.float32).reshape(-1)

        tx_t = torch.from_numpy(tx.astype(np.float32)).to(self.device)
        tw_t = torch.from_numpy(tw.astype(np.float32)).to(self.device)
        tz_t = torch.from_numpy(tz.astype(np.float32)).to(self.device)
        A_t = torch.from_numpy(A).to(self.device)

        self.a_head.eval()
        self.y_head.eval()
        with torch.no_grad():
            s_logits = self.a_head(torch.cat([tx_t, tz_t], dim=1)).cpu().numpy()
            t_obs_std = self.y_head(torch.cat([tx_t, tw_t], dim=1), A_t).cpu().numpy()
            ones = torch.ones_like(A_t)
            zeros = torch.zeros_like(A_t)
            t1_std = self.y_head(torch.cat([tx_t, tw_t], dim=1), ones).cpu().numpy()
            t0_std = self.y_head(torch.cat([tx_t, tw_t], dim=1), zeros).cpu().numpy()

        t_obs = self._destandardize_y(t_obs_std)
        t1 = self._destandardize_y(t1_std)
        t0 = self._destandardize_y(t0_std)
        return s_logits, t_obs, t1, t0

    def _split_latent(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        cfg = self.config
        dx, dw, dz = cfg.latent_x_dim, cfg.latent_w_dim, cfg.latent_z_dim
        tx = U[:, :dx]
        tw = U[:, dx : dx + dw]
        tz = U[:, dx + dw : dx + dw + dz]
        tn: Optional[np.ndarray] = None
        if cfg.use_noise_in_latent and U.shape[1] > dx + dw + dz:
            tn = U[:, dx + dw + dz :]
        return tx, tw, tz, tn

    def _dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        cfg = self.config
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        weak_iv = bool(getattr(self, "_weak_iv_flag", False))
        dr_mode = "weak_iv_conservative" if weak_iv else "default"
        stats: Dict[str, list] = {
            k: []
            for k in [
                "e_min",
                "e_max",
                "e_q01",
                "e_q05",
                "e_q95",
                "e_q99",
                "clip_used",
                "cap_used",
                "overlap_score",
                "frac_e_clipped",
                "ipw_abs_max_raw",
                "ipw_abs_max_postclip_precap",
                "ipw_abs_max_capped",
                "frac_ipw_capped",
                "frac_ipw_capped_raw",
            ]
        }

        def _quantiles(x: np.ndarray) -> Dict[str, float]:
            if x.size == 0:
                return {"min": np.nan, "max": np.nan, "q01": np.nan, "q05": np.nan, "q95": np.nan, "q99": np.nan}
            return {
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "q01": float(np.quantile(x, 0.01)),
                "q05": float(np.quantile(x, 0.05)),
                "q95": float(np.quantile(x, 0.95)),
                "q99": float(np.quantile(x, 0.99)),
            }

        kf = KFold(n_splits=cfg.n_splits_dr, shuffle=True, random_state=cfg.seed)
        psi = np.zeros_like(Y, dtype=float)
        clip = float(max(cfg.clip_prop_radr, cfg.clip_prop))
        ipw_cap = cfg.ipw_cap_radr if getattr(cfg, "ipw_cap_radr", None) is not None else cfg.ipw_cap
        if weak_iv:
            clip = max(clip, 0.05)
            if ipw_cap is not None:
                ipw_cap = min(float(ipw_cap), 10.0)

        for tr, te in kf.split(U):
            A_tr, A_te = A[tr], A[te]
            Y_tr, Y_te = Y[tr], Y[te]

            s_tr, t_obs_tr, t1_tr, t0_tr = self._head_features(U[tr], A_tr)
            s_te, t_obs_te, t1_te, t0_te = self._head_features(U[te], A_te)

            tx_tr, tw_tr, tz_tr, tn_tr = self._split_latent(U[tr])
            tx_te, tw_te, tz_te, tn_te = self._split_latent(U[te])

            s_tr_clipped = np.clip(s_tr, -10, 10)
            s_te_clipped = np.clip(s_te, -10, 10)
            prop_feats_tr = [s_tr_clipped.reshape(-1, 1), tx_tr, tz_tr]
            prop_feats_te = [s_te_clipped.reshape(-1, 1), tx_te, tz_te]
            if tn_tr is not None:
                prop_feats_tr.append(tn_tr)
                prop_feats_te.append(tn_te)
            Xp_tr = np.column_stack(prop_feats_tr)
            Xp_te = np.column_stack(prop_feats_te)
            if cfg.standardize_nuisance:
                scaler_prop = StandardScaler().fit(Xp_tr)
                Xp_tr = scaler_prop.transform(Xp_tr)
                Xp_te = scaler_prop.transform(Xp_te)
            if np.unique(A_tr).size < 2:
                e_raw = np.full_like(A_te, float(np.mean(A_tr)) if len(A_tr) else 0.5, dtype=float)
            else:
                prop = LogisticRegression(max_iter=2000, solver="lbfgs", C=cfg.propensity_logreg_C)
                prop.fit(Xp_tr, A_tr)
                e_raw = prop.predict_proba(Xp_te)[:, 1]
            if cfg.propensity_shrinkage > 0:
                prior = float(np.mean(A_tr)) if len(A_tr) else 0.5
                e_raw = (1 - cfg.propensity_shrinkage) * e_raw + cfg.propensity_shrinkage * prior

            qs = _quantiles(e_raw)
            clip_use = clip
            cap_use = ipw_cap
            e_hat = np.clip(e_raw, clip_use, 1 - clip_use)

            stats["clip_used"].append(float(clip_use))
            stats["cap_used"].append(float(cap_use) if cap_use is not None else np.nan)
            eps = max(clip_use, 0.05)
            overlap_score = float(np.mean((e_raw > eps) & (e_raw < 1 - eps))) if e_raw.size else np.nan
            stats["overlap_score"].append(overlap_score)
            stats["frac_e_clipped"].append(float(np.mean((e_raw < clip_use) | (e_raw > 1 - clip_use))))
            for k, v in qs.items():
                stats[f"e_{k}"].append(v)

            def _outcome_features(a_vec, tx_block, tw_block, t_val, tn_block):
                feats = [
                    a_vec.reshape(-1, 1),
                    tx_block,
                    tw_block,
                    t_val.reshape(-1, 1),
                    (a_vec * t_val).reshape(-1, 1),
                    (a_vec[:, None] * tx_block),
                    (a_vec[:, None] * tw_block),
                ]
                if tn_block is not None:
                    feats.append(tn_block)
                    feats.append(a_vec[:, None] * tn_block)
                return np.column_stack(feats)

            Xo_tr = _outcome_features(A_tr, tx_tr, tw_tr, t_obs_tr, tn_tr)
            Xo_te = _outcome_features(A_te, tx_te, tw_te, t_obs_te, tn_te)
            if cfg.standardize_nuisance:
                scaler_out = StandardScaler().fit(Xo_tr)
                Xo_tr = scaler_out.transform(Xo_tr)
                Xo_te = scaler_out.transform(Xo_te)

            if len(A_tr) == 0:
                out_model: Ridge | DummyRegressor = DummyRegressor(strategy="mean")
                out_model.fit(np.zeros((1, Xo_tr.shape[1])), [0.0])
            else:
                out_model = Ridge(alpha=cfg.ridge_alpha)
                out_model.fit(Xo_tr, Y_tr)

            X1 = _outcome_features(np.ones_like(A_te), tx_te, tw_te, t1_te, tn_te)
            X0 = _outcome_features(np.zeros_like(A_te), tx_te, tw_te, t0_te, tn_te)
            if cfg.standardize_nuisance:
                X1 = scaler_out.transform(X1)
                X0 = scaler_out.transform(X0)

            m_hat = out_model.predict(Xo_te)
            m1 = out_model.predict(X1)
            m0 = out_model.predict(X0)

            eps_raw = 1e-6
            e_safe = np.clip(e_raw, eps_raw, 1 - eps_raw)
            w_raw = (A_te - e_safe) / (e_safe * (1 - e_safe))

            stats["ipw_abs_max_raw"].append(float(np.max(np.abs(w_raw))) if w_raw.size else np.nan)
            # post-clip (propensity) but pre-cap weights
            w = (A_te - e_hat) / (e_hat * (1 - e_hat))
            stats["ipw_abs_max_postclip_precap"].append(float(np.max(np.abs(w))) if w.size else np.nan)

            if ipw_cap and ipw_cap > 0:
                cap_val = float(ipw_cap)
                if cfg.ipw_cap_quantile is not None and 0 < cfg.ipw_cap_quantile < 1 and w.size:
                    cap_val = min(cap_val, float(np.quantile(np.abs(w), cfg.ipw_cap_quantile)))
                w_capped = np.clip(w, -cap_val, cap_val)
            else:
                w_capped = w
            stats["ipw_abs_max_capped"].append(float(np.max(np.abs(w_capped))) if w_capped.size else np.nan)
            if ipw_cap and ipw_cap > 0:
                stats["frac_ipw_capped"].append(
                    float(np.mean(np.abs(w) >= float(ipw_cap))) if w.size else np.nan
                )
                stats["frac_ipw_capped_raw"].append(
                    float(np.mean(np.abs(w_raw) >= float(ipw_cap))) if w_raw.size else np.nan
                )
            else:
                stats["frac_ipw_capped"].append(np.nan)
                stats["frac_ipw_capped_raw"].append(np.nan)

            psi[te] = m1 - m0 + w_capped * (Y_te - m_hat)

        if not hasattr(self, "training_diagnostics"):
            self.training_diagnostics = {}
        if stats:
            agg = {k: float(np.nanmean(v)) for k, v in stats.items() if v}
            self.training_diagnostics.update({f"radr_{k}": v for k, v in agg.items()})
            self.training_diagnostics["dr_mode"] = dr_mode

        return float(np.mean(psi))


__all__ = [
    "IVAPCIV33TheoryConfig",
    "IVAPCIv33TheoryHierEstimator",
    "IVAPCIv33TheoryHierRADREstimator",
]
