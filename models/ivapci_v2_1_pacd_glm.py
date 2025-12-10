"""IVAPCI v2.1 encoder + PACD-T partition + leafwise DR-GLM estimator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold

from . import BaseCausalEstimator
from .ivapci_v21 import IVAPCIConfig, IVAPCIv21Estimator
from .pacdt_v30 import PACDTree


@dataclass
class IVAPCIPACDTGLMConfig:
    """Configuration for IVAPCI + PACD-T leafwise DR-GLM estimator."""

    ivapci_config: Optional[IVAPCIConfig] = None
    tree_max_depth: int = 3
    tree_min_leaf_size: int = 120
    min_leaf_for_dr: int = 100
    n_splits: int = 2
    seed: int = 42


def _dr_scores_glm_local(
    U: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Compute DR-GLM scores within a single leaf using cross-fitting.

    If a leaf lacks both treatment arms or is too small to cross-fit, the scores
    gracefully fall back to a single-model DR calculation rather than raising
    errors, ensuring callers can choose to keep global estimates instead.
    """

    U = np.asarray(U)
    A = np.asarray(A).reshape(-1)
    Y = np.asarray(Y).reshape(-1)

    n = U.shape[0]
    psi = np.zeros(n, dtype=float)

    # If there is only one treatment arm or too few samples, return zeros so
    # callers can fall back to global estimates without crashing.
    if np.unique(A).size < 2 or n < 2:
        return psi

    if n_splits <= 1 or n < 2 * n_splits:
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(U, A)
        e_hat = clf.predict_proba(U)[:, 1].clip(1e-3, 1 - 1e-3)

        X_all = np.column_stack([A, U])
        reg = LinearRegression().fit(X_all, Y)
        X1 = np.column_stack([np.ones_like(A), U])
        X0 = np.column_stack([np.zeros_like(A), U])
        m1_hat = reg.predict(X1)
        m0_hat = reg.predict(X0)

        psi[:] = (
            m1_hat
            - m0_hat
            + A * (Y - m1_hat) / e_hat
            - (1 - A) * (Y - m0_hat) / (1 - e_hat)
        )
        return psi

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(U):
        U_tr, U_te = U[train_idx], U[test_idx]
        A_tr, A_te = A[train_idx], A[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        # Extreme imbalance inside a fold: skip and rely on other folds/global
        if np.unique(A_tr).size < 2:
            continue

        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(U_tr, A_tr)
        e_hat = clf.predict_proba(U_te)[:, 1].clip(1e-3, 1 - 1e-3)

        X_tr = np.column_stack([A_tr, U_tr])
        reg = LinearRegression().fit(X_tr, Y_tr)

        X1 = np.column_stack([np.ones_like(A_te), U_te])
        X0 = np.column_stack([np.zeros_like(A_te), U_te])
        m1_hat = reg.predict(X1)
        m0_hat = reg.predict(X0)

        psi[test_idx] = (
            m1_hat
            - m0_hat
            + A_te * (Y_te - m1_hat) / e_hat
            - (1 - A_te) * (Y_te - m0_hat) / (1 - e_hat)
        )

    return psi


class IVAPCIPACDTGLMEstimator(BaseCausalEstimator):
    """IVAPCI v2.1 encoder + PACD-T partition + leafwise DR-GLM."""

    def __init__(self, config: Optional[IVAPCIPACDTGLMConfig] = None) -> None:
        self.config = config or IVAPCIPACDTGLMConfig()
        if self.config.ivapci_config is None:
            self.config.ivapci_config = IVAPCIConfig(device="cpu")

        self._ivapci_encoder: Optional[IVAPCIv21Estimator] = None
        self._tree: Optional[PACDTree] = None
        self._is_fit = False

    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        if PACDTree is None:
            raise ImportError("PACDTree is unavailable; ensure pacdt_v3_0 exposes PACDTree.")

        X_all = np.asarray(X_all, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32).reshape(-1)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)

        self._ivapci_encoder = IVAPCIv21Estimator(self.config.ivapci_config)
        self._ivapci_encoder.fit(X_all, A, Y)
        U_train = self._ivapci_encoder.get_latent(X_all)

        self._tree = PACDTree(
            max_depth=self.config.tree_max_depth,
            min_leaf_size=self.config.tree_min_leaf_size,
            seed=self.config.seed,
            include_treatment=True,
        )
        self._tree.fit(U_train, A, Y)
        self._is_fit = True

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        if not self._is_fit or self._ivapci_encoder is None or self._tree is None:
            raise RuntimeError("Estimator must be fit before estimating ATE.")

        X_all = np.asarray(X_all, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32).reshape(-1)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)

        U_hat = self._ivapci_encoder.get_latent(X_all)
        leaf_ids = self._tree.apply(U_hat, A)

        # Global DR-GLM as a stable fallback for small or imbalanced leaves
        psi_global = _dr_scores_glm_local(
            U_hat,
            A,
            Y,
            n_splits=self.config.n_splits,
            seed=self.config.seed,
        )
        psi = psi_global.copy()

        for leaf in np.unique(leaf_ids):
            idx = np.where(leaf_ids == leaf)[0]
            if idx.size < self.config.min_leaf_for_dr:
                # keep global estimates for small leaves
                continue

            psi_leaf = _dr_scores_glm_local(
                U_hat[idx],
                A[idx],
                Y[idx],
                n_splits=self.config.n_splits,
                seed=self.config.seed + int(leaf),
            )

            # Only override when we obtained meaningful local scores
            if np.any(psi_leaf):
                psi[idx] = psi_leaf

        return float(psi.mean())

    def get_latent(self, X_all: np.ndarray) -> np.ndarray:
        if not self._is_fit or self._ivapci_encoder is None:
            raise RuntimeError("Estimator must be fit before get_latent.")
        return self._ivapci_encoder.get_latent(X_all)


__all__ = ["IVAPCIPACDTGLMConfig", "IVAPCIPACDTGLMEstimator"]
