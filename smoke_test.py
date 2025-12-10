"""Lightweight smoke test for simulators, estimators, and diagnostics.

Runs a tiny simulation, trains a baseline and IVAPCI estimator with
shortened training, and computes a proxy strength diagnostic to ensure
core components execute end-to-end.
"""
from __future__ import annotations

import numpy as np

from diagnostics import proxy_strength_score
from models.baselines import NaiveEstimator
from models.ivapci_v21 import IVAPCIConfig, IVAPCIv21Estimator
from simulators import simulate_scenario


def main() -> None:
    data = simulate_scenario("EASY-linear-weak", n=40, seed=0)
    X_all = np.concatenate([data["X"], data["W"], data["Z"]], axis=1)

    naive = NaiveEstimator()
    naive.fit(X_all, data["A"], data["Y"])
    ate_naive = naive.estimate_ate(X_all, data["A"], data["Y"])

    cfg = IVAPCIConfig(
        epochs=5,
        batch_size=32,
        latent_dim=2,
        encoder_hidden=(16,),
        decoder_hidden=(16,),
        a_hidden=(8,),
        y_hidden=(8,),
    )
    ivapci = IVAPCIv21Estimator(cfg)
    ivapci.fit(X_all, data["A"], data["Y"])
    ate_ivapci = ivapci.estimate_ate(X_all, data["A"], data["Y"])
    latent = ivapci.get_latent(X_all)

    prox = proxy_strength_score(
        data["U"], data["X"], data["W"], data["Z"], data["A"], data["Y"], n_splits=2
    )

    print("Naive ATE:", ate_naive)
    print("IVAPCI ATE:", ate_ivapci)
    print("Latent shape:", latent.shape)
    print("Proxy strength diagnostic:", prox)


if __name__ == "__main__":
    main()
