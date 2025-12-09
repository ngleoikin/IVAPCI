"""Baseline causal estimators matching the unified benchmark interface.

The estimators below implement the behavior described in
``docs/pacd_benchmark_design.md`` and provide lightweight implementations
suitable for simulation benchmarks:

* ``NaiveEstimator``: difference in means.
* ``DRGLMEstimator``: cross-fitted doubly robust estimator with GLM base
  learners (logistic regression for propensity, linear regression for
  outcomes).
* ``DRRFEstimator``: same DR structure but with random forest base learners.
* ``OracleUEstimator``: uses the provided ``X_all`` as the true confounder and
  applies the same DR machinery.

Each estimator adheres to the ``BaseCausalEstimator`` interface.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from . import BaseCausalEstimator


class NaiveEstimator(BaseCausalEstimator):
    """Naive difference-in-means estimator."""

    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        self._ate = float(Y[A == 1].mean() - Y[A == 0].mean())

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        return getattr(self, "_ate", float(Y[A == 1].mean() - Y[A == 0].mean()))

    def get_latent(self, X_all: np.ndarray) -> Optional[np.ndarray]:
        return None


class DRGLMEstimator(BaseCausalEstimator):
    """Doubly-robust estimator with generalized linear models."""

    def __init__(self, n_splits: int = 2, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        self._ate = float(self._dr_ate(X_all, A, Y))

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        return getattr(self, "_ate", float(self._dr_ate(X_all, A, Y)))

    def get_latent(self, X_all: np.ndarray) -> np.ndarray:
        return np.asarray(X_all)

    def _dr_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        U = np.asarray(X_all)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        psi = np.zeros_like(Y, dtype=float)
        for train_idx, test_idx in kf.split(U):
            U_train, U_test = U[train_idx], U[test_idx]
            A_train, A_test = A[train_idx], A[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            prop_model = LogisticRegression(max_iter=1000)
            prop_model.fit(U_train, A_train)
            e_hat = prop_model.predict_proba(U_test)[:, 1]
            e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)

            m1 = LinearRegression()
            m0 = LinearRegression()
            m1.fit(U_train[A_train == 1], Y_train[A_train == 1])
            m0.fit(U_train[A_train == 0], Y_train[A_train == 0])

            m1_hat = m1.predict(U_test)
            m0_hat = m0.predict(U_test)
            m_hat = np.where(A_test == 1, m1_hat, m0_hat)

            psi[test_idx] = (
                (A_test - e_hat) / (e_hat * (1 - e_hat)) * (Y_test - m_hat)
                + m1_hat
                - m0_hat
            )
        return float(np.mean(psi))


class DRRFEstimator(BaseCausalEstimator):
    """Doubly-robust estimator with random forests."""

    def __init__(self, n_estimators: int = 200, n_splits: int = 2, seed: int = 42):
        self.n_estimators = n_estimators
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        self._ate = float(self._dr_ate(X_all, A, Y))

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        return getattr(self, "_ate", float(self._dr_ate(X_all, A, Y)))

    def get_latent(self, X_all: np.ndarray) -> np.ndarray:
        return np.asarray(X_all)

    def _dr_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        U = np.asarray(X_all)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        psi = np.zeros_like(Y, dtype=float)
        for train_idx, test_idx in kf.split(U):
            U_train, U_test = U[train_idx], U[test_idx]
            A_train, A_test = A[train_idx], A[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            prop_model = RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=self.seed
            )
            prop_model.fit(U_train, A_train)
            e_hat = prop_model.predict_proba(U_test)[:, 1]
            e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)

            m1 = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=self.seed
            )
            m0 = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=self.seed
            )
            m1.fit(U_train[A_train == 1], Y_train[A_train == 1])
            m0.fit(U_train[A_train == 0], Y_train[A_train == 0])

            m1_hat = m1.predict(U_test)
            m0_hat = m0.predict(U_test)
            m_hat = np.where(A_test == 1, m1_hat, m0_hat)

            psi[test_idx] = (
                (A_test - e_hat) / (e_hat * (1 - e_hat)) * (Y_test - m_hat)
                + m1_hat
                - m0_hat
            )
        return float(np.mean(psi))


class OracleUEstimator(BaseCausalEstimator):
    """Oracle estimator using true confounder as input features."""

    def __init__(self, n_splits: int = 2, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        self._ate = float(self._dr_ate(X_all, A, Y))

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        return getattr(self, "_ate", float(self._dr_ate(X_all, A, Y)))

    def get_latent(self, X_all: np.ndarray) -> np.ndarray:
        """Return the provided confounder proxy (assumed to be true U in sims)."""

        return np.asarray(X_all)

    def _dr_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        U = np.asarray(X_all)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        psi = np.zeros_like(Y, dtype=float)
        for train_idx, test_idx in kf.split(U):
            U_train, U_test = U[train_idx], U[test_idx]
            A_train, A_test = A[train_idx], A[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            prop_model = LogisticRegression(max_iter=1000)
            prop_model.fit(U_train, A_train)
            e_hat = prop_model.predict_proba(U_test)[:, 1]
            e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)

            m1 = LinearRegression()
            m0 = LinearRegression()
            m1.fit(U_train[A_train == 1], Y_train[A_train == 1])
            m0.fit(U_train[A_train == 0], Y_train[A_train == 0])

            m1_hat = m1.predict(U_test)
            m0_hat = m0.predict(U_test)
            m_hat = np.where(A_test == 1, m1_hat, m0_hat)

            psi[test_idx] = (
                (A_test - e_hat) / (e_hat * (1 - e_hat)) * (Y_test - m_hat)
                + m1_hat
                - m0_hat
            )
        return float(np.mean(psi))


__all__ = [
    "NaiveEstimator",
    "DRGLMEstimator",
    "DRRFEstimator",
    "OracleUEstimator",
]
