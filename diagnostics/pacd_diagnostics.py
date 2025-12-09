"""Diagnostics interface matching pacd_benchmark_design specifications."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import auc, roc_curve, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


def _concat_features(X: Optional[np.ndarray], W: Optional[np.ndarray], Z: Optional[np.ndarray]) -> np.ndarray:
    parts = [p for p in [X, W, Z] if p is not None]
    if not parts:
        raise ValueError("At least one of X, W, Z must be provided.")
    return np.concatenate(parts, axis=1)


def _cross_fitted_predictions(model, X: np.ndarray, y: np.ndarray, n_splits: int, random_state: int) -> np.ndarray:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in kf.split(X):
        est = clone(model)
        est.fit(X[train_idx], y[train_idx])
        fold_pred = est.predict(X[test_idx])
        preds[test_idx] = fold_pred
    return preds


def estimate_residual_risk(
    X: Optional[np.ndarray],
    W: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    A: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> Dict[str, float]:
    """Estimate residual correlation risk following the design specification."""

    V = _concat_features(X, W, Z)
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    reg = RandomForestRegressor(n_estimators=200, random_state=random_state)

    e_hat = _cross_fitted_predictions(clf, V, A, n_splits=n_splits, random_state=random_state)
    m_hat = _cross_fitted_predictions(reg, V, Y, n_splits=n_splits, random_state=random_state)

    r_A = A - e_hat
    r_Y = Y - m_hat

    corr = float(np.corrcoef(r_A, r_Y)[0, 1]) if np.std(r_A) > 0 and np.std(r_Y) > 0 else 0.0

    try:
        auc_A = float(roc_auc_score(A, e_hat))
    except Exception:
        # roc_auc_score fails if only one class is present
        fpr, tpr, _ = roc_curve(A, e_hat)
        auc_A = float(auc(fpr, tpr))

    r2_Y = float(r2_score(Y, m_hat)) if np.std(Y) > 0 else 0.0

    return {
        "resid_score": abs(corr),
        "resid_corr": corr,
        "auc_A": auc_A,
        "r2_Y": r2_Y,
    }


def proxy_strength_score(
    U_true: np.ndarray,
    X: Optional[np.ndarray],
    W: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    A: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> Dict[str, float]:
    """Compute proxy strength score as the average of R2/AUC components."""

    V = _concat_features(X, W, Z)
    # Predict U_true (multi-output regression)
    reg_u = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=random_state))
    u_hat = _cross_fitted_predictions(reg_u, V, U_true, n_splits=n_splits, random_state=random_state)
    r2_U = float(r2_score(U_true, u_hat)) if np.all(np.std(U_true, axis=0) > 0) else 0.0

    # Predict Y
    reg_y = RandomForestRegressor(n_estimators=200, random_state=random_state)
    y_hat = _cross_fitted_predictions(reg_y, V, Y, n_splits=n_splits, random_state=random_state)
    r2_Y = float(r2_score(Y, y_hat)) if np.std(Y) > 0 else 0.0

    # Predict A
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    a_hat = _cross_fitted_predictions(clf, V, A, n_splits=n_splits, random_state=random_state)
    try:
        auc_A = float(roc_auc_score(A, a_hat))
    except Exception:
        fpr, tpr, _ = roc_curve(A, a_hat)
        auc_A = float(auc(fpr, tpr))

    components = [max(0.0, min(1.0, r2_U)), max(0.0, min(1.0, r2_Y)), max(0.0, min(1.0, auc_A))]
    proxy_score = float(np.mean(components))

    return {
        "proxy_score": proxy_score,
        "r2_U": r2_U,
        "r2_Y": r2_Y,
        "auc_A": auc_A,
    }


def proximal_condition_number(
    X: Optional[np.ndarray],
    W: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute proximal condition number using proxy matrices."""

    if Z is not None and W is not None:
        M = np.concatenate([Z, W], axis=1)
    elif Z is not None:
        M = Z
    elif W is not None:
        M = W
    elif X is not None:
        M = X
    else:
        raise ValueError("At least one proxy matrix must be provided.")

    scaler = StandardScaler()
    M_std = scaler.fit_transform(M)
    cov = np.cov(M_std, rowvar=False)
    s = np.linalg.svd(cov, compute_uv=False)
    s_filtered = s[s > eps]
    if s_filtered.size == 0:
        return {"prox_cond_score": np.inf, "cond": np.inf, "s_min": 0.0, "s_max": 0.0}

    s_min, s_max = float(np.min(s_filtered)), float(np.max(s_filtered))
    cond = float(s_max / s_min) if s_min > 0 else np.inf

    return {
        "prox_cond_score": cond,
        "cond": cond,
        "s_min": s_min,
        "s_max": s_max,
    }


def _linear_r2(U_true: np.ndarray, U_hat: Optional[np.ndarray]) -> Tuple[float, np.ndarray]:
    if U_hat is None:
        return 0.0, np.array([])

    U_true = np.asarray(U_true)
    U_hat = np.asarray(U_hat)
    if U_hat.ndim == 1:
        U_hat = U_hat.reshape(-1, 1)

    XtX_inv = np.linalg.pinv(U_hat.T @ U_hat)
    coefs = XtX_inv @ U_hat.T @ U_true
    preds = U_hat @ coefs
    ss_res = np.sum((U_true - preds) ** 2, axis=0)
    ss_tot = np.sum((U_true - U_true.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
    return float(np.mean(r2)), r2


def extract_confounding_subspace(
    U_true: np.ndarray,
    U_hat_ivapci: Optional[np.ndarray],
    U_hat_pacdt: Optional[np.ndarray],
    scenario: str,
    rep: int,
    outdir: str = "subspace_plots",
) -> Dict[str, Dict[str, object]]:
    """Align latent subspaces and produce PCA visualizations."""

    os.makedirs(outdir, exist_ok=True)

    ivapci_mean_r2, ivapci_all = _linear_r2(U_true, U_hat_ivapci)
    pacdt_mean_r2, pacdt_all = _linear_r2(U_true, U_hat_pacdt)

    plot_path = ""
    if U_true.shape[0] >= 2:
        pca = PCA(n_components=2)
        true_proj = pca.fit_transform(U_true)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].scatter(true_proj[:, 0], true_proj[:, 1], s=10, alpha=0.7)
        axes[0].set_title("U_true (PCA)")

        if U_hat_ivapci is not None:
            iv_proj = pca.transform(U_hat_ivapci)
            axes[1].scatter(iv_proj[:, 0], iv_proj[:, 1], s=10, alpha=0.7, color="tab:orange")
        axes[1].set_title("IVAPCI latent (PCA basis)")

        if U_hat_pacdt is not None:
            pac_proj = pca.transform(U_hat_pacdt)
            axes[2].scatter(pac_proj[:, 0], pac_proj[:, 1], s=10, alpha=0.7, color="tab:green")
        axes[2].set_title("PACD-T latent (PCA basis)")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plot_path = os.path.join(outdir, f"subspace_{scenario}_rep{rep}.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

    return {
        "ivapci": {
            "mean_best_r2": float(ivapci_mean_r2),
            "all_r2": ivapci_all,
            "plot_path": plot_path,
        },
        "pacdt": {
            "mean_best_r2": float(pacdt_mean_r2),
            "all_r2": pacdt_all,
            "plot_path": plot_path,
        },
    }


__all__ = [
    "estimate_residual_risk",
    "proxy_strength_score",
    "proximal_condition_number",
    "extract_confounding_subspace",
]
