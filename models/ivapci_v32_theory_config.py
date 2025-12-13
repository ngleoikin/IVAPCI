"""Theory-guided wrapper config for IVAPCI v3.2 hierarchical encoders."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ivapci_v32_hierarchical import IVAPCIV32HierConfig, IVAPCIv32HierRADREstimator


@dataclass
class TheoryGuidedIVAPCIV32Config(IVAPCIV32HierConfig):
    """Auto-tuned config derived from sample size heuristics.

    The heuristics follow the bias–variance and information bottleneck
    considerations discussed in the theory notes:
    * latent dims grow slowly with n
    * orthogonality / p-adic / consistency regularization strengthen mildly with n
    * adversarial weights relax slightly for small n to avoid collapse
    * training epochs scale gently with n
    """

    scale_factor: float = 1.0

    def auto_tune_by_n(self, n_samples: int) -> None:
        # 1) latent dims: d ≈ 4 * log(1 + n/100)
        base = max(4.0, 4.0 * np.log(1.0 + n_samples / 100.0))
        base *= self.scale_factor
        d_total = int(round(base))
        d_total = max(4, min(d_total, 64))

        self.latent_x_dim = max(2, d_total // 4)
        self.latent_w_dim = max(2, d_total // 4)
        self.latent_z_dim = max(2, d_total // 4)
        self.latent_n_dim = max(2, d_total // 4)

        # 2) regularizers scale with sqrt(n)
        scale = np.sqrt(max(1.0, n_samples / 1000.0))
        self.lambda_ortho = float(0.005 * scale)
        self.gamma_padic = float(1e-3 * scale)
        self.lambda_consistency = float(0.05 * scale)

        # 3) adversaries relax for small n
        inv_scale = 1.0 / scale
        self.gamma_adv_w = float(0.1 * inv_scale)
        self.gamma_adv_z = float(0.1 * inv_scale)
        self.gamma_adv_n = float(0.1 * inv_scale)

        # 4) epochs scale mildly with n
        self.epochs_pretrain = int(min(60, 20 + n_samples / 200))
        self.epochs_main = int(min(200, 60 + n_samples / 100))


def make_theory_guided_v32_estimator(
    x_dim: int,
    w_dim: int,
    z_dim: int,
    n_samples: int,
    *,
    device: str = "cpu",
) -> IVAPCIv32HierRADREstimator:
    """Construct a theory-guided v3.2 RADR estimator tuned to sample size."""

    cfg = TheoryGuidedIVAPCIV32Config(
        x_dim=x_dim,
        w_dim=w_dim,
        z_dim=z_dim,
        device=device,
    )
    cfg.auto_tune_by_n(n_samples)
    return IVAPCIv32HierRADREstimator(config=cfg)


__all__ = [
    "TheoryGuidedIVAPCIV32Config",
    "make_theory_guided_v32_estimator",
]

