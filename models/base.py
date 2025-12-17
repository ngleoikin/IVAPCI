"""Unified causal estimator interface for PACD/IVAPCI benchmarks."""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np


class BaseCausalEstimator(abc.ABC):
    """Abstract base class for causal estimators used in the benchmarks."""

    @abc.abstractmethod
    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        """Fit model using provided features, treatment, and outcome."""

    @abc.abstractmethod
    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        """Estimate average treatment effect on the supplied dataset."""

    def get_latent(self, X_all: np.ndarray) -> Optional[np.ndarray]:
        """Return learned latent representation, if available."""
        return None


__all__ = ["BaseCausalEstimator"]
