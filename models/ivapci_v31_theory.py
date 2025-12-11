"""Theory-guided IVAPCI v3.1 variant with information-aware standardization and RADR.

This estimator reuses the PACD-enhanced v3.1 encoder and RADR nuisance
calibration, while adding two lightweight improvements inspired by the
information-theoretic notes:

* Protect near-constant proxy dimensions during standardization to avoid
  discarding deterministic signal channels.
* Record a reconstruction-based information-loss proxy after training so that
  downstream scripts can inspect whether representation collapse is likely to
  dominate estimation error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .ivapci_v31_pacd_encoder import (
    IVAPCIV31Config,
    IVAPCIv31RADREstimator,
    _apply_standardize,
    _standardize,
)


def _info_aware_standardize(
    train: np.ndarray, min_std: float = 1e-2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize while keeping track of protected low-variance dimensions.

    Low-variance channels may still carry deterministic proxy information; we
    clamp their scale instead of replacing it with 1.0 and return a mask so
    diagnostics can flag how many features were protected.
    """

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    protected = std < min_std
    std = np.maximum(std, min_std)
    return (train - mean) / std, mean, std, protected.squeeze(0)


class InformationLossMonitor:
    """Rudimentary proxy for representation information loss.

    We approximate the information gap with the reconstruction MSE of the
    proxy decoder relative to the variance of standardized proxies. This keeps
    the computation lightweight and aligns with the Theorem 1 guidance that
    higher reconstruction error implies larger information loss.
    """

    def estimate(self, model: IVAPCIv31RADREstimator, X_all: np.ndarray) -> dict:
        X_std = _apply_standardize(X_all.astype(np.float32), model._x_mean, model._x_std)
        with torch.no_grad():
            X_t = torch.from_numpy(X_std).to(model.device)
            h = model.shared(X_t)
            mu_c, logvar_c = model.c_head(h)
            mu_n, logvar_n = model.n_head(h)
            z_c = mu_c  # deterministic mean representation
            z_n = mu_n
            recon = model.proxy_decoder(torch.cat([z_c, z_n], dim=1)).cpu().numpy()
        recon_mse = float(np.mean((recon - X_std) ** 2))
        variance = float(np.var(X_std) + 1e-8)
        delta_I = recon_mse / variance if variance > 0 else recon_mse
        return {"recon_mse": recon_mse, "info_loss_proxy": delta_I}


@dataclass
class IVAPCIV31TheoryConfig(IVAPCIV31Config):
    min_std: float = 1e-2


class IVAPCIv31TheoryRADREstimator(IVAPCIv31RADREstimator):
    """v3.1 encoder + RADR with information-aware preprocessing and diagnostics."""

    def __init__(self, config: Optional[IVAPCIV31TheoryConfig] = None):
        super().__init__(config=config or IVAPCIV31TheoryConfig())
        self.info_monitor = InformationLossMonitor()
        self._protected_mask: Optional[np.ndarray] = None
        self.training_diagnostics: dict = {}

    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        # Apply theory-aware standardization before delegating to the base fit.
        X_all = np.asarray(X_all, dtype=np.float32)
        X_std, self._x_mean, self._x_std, protected = _info_aware_standardize(
            X_all, min_std=self.config.min_std
        )
        self._protected_mask = protected
        # Reuse the parent training pipeline by temporarily calling the shared
        # standardizer signature.
        super().fit(X_std, A, Y)

        # After training, record reconstruction-based information-loss signals.
        self.training_diagnostics = self.info_monitor.estimate(self, X_all)
        self.training_diagnostics["protected_features"] = int(np.sum(protected))

    def get_training_diagnostics(self) -> dict:
        return self.training_diagnostics


__all__ = ["IVAPCIV31TheoryConfig", "IVAPCIv31TheoryRADREstimator"]

