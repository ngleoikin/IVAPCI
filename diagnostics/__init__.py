"""Diagnostics package for PACD/IVAPCI benchmarks."""

from .pacd_diagnostics import (
    estimate_residual_risk,
    proxy_strength_score,
    proximal_condition_number,
    extract_confounding_subspace,
)

__all__ = [
    "estimate_residual_risk",
    "proxy_strength_score",
    "proximal_condition_number",
    "extract_confounding_subspace",
]
