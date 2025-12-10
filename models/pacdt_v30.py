"""PACD-T v3.0 implementation following docs/PACD_T_v3.0_spec.md."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from . import BaseCausalEstimator


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.coeff * grad_output, None


def _grl(x: torch.Tensor, coeff: float) -> torch.Tensor:
    return _GradientReversal.apply(x, coeff)


def _standardize(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, mean, std


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim_c: int, latent_dim_n: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
        last = input_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        self.net = nn.Sequential(*layers)
        self.mu_c = nn.Linear(last, latent_dim_c)
        self.logvar_c = nn.Linear(last, latent_dim_c)
        self.mu_n = nn.Linear(last, latent_dim_n)
        self.logvar_n = nn.Linear(last, latent_dim_n)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_c(h), self.logvar_c(h), self.mu_n(h), self.logvar_n(h)


class _ProxyDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
        last = latent_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _HeadA(nn.Module):
    def __init__(self, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
        last = latent_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class _HeadY(nn.Module):
    def __init__(self, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
        last = latent_dim + 1
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a.unsqueeze(-1)], dim=1)).squeeze(-1)


class _AdversaryA(nn.Module):
    def __init__(self, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
        last = latent_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def _p_adic_loss(u_c: torch.Tensor, p: int = 2, n_triplets: int = 32, eps: float = 1e-6) -> torch.Tensor:
    """Ultrametric regularizer on the causal latent branch.

    Implements a lightweight version of the p-adic-inspired loss described in
    docs/PACD_T_v3.0_spec.md by sampling triplets and penalizing violations of
    the ultrametric inequality.
    """

    bsz, dim = u_c.shape
    if bsz < 3:
        return torch.tensor(0.0, device=u_c.device, dtype=u_c.dtype)

    std = torch.std(u_c, dim=0, unbiased=False)
    step = torch.where(std > eps, std * 0.5, torch.ones_like(std))
    quant = torch.floor(u_c / step).to(torch.int64)

    n_triplets = min(n_triplets, bsz // 3)
    if n_triplets == 0:
        return torch.tensor(0.0, device=u_c.device, dtype=u_c.dtype)
    idx = torch.randperm(bsz, device=u_c.device)[: n_triplets * 3].view(n_triplets, 3)

    def _valuation(n: torch.Tensor) -> torch.Tensor:
        n = torch.abs(n)
        r = torch.zeros_like(n)
        mask = n != 0
        while mask.any():
            divisible = (n % p == 0) & mask
            r = r + divisible.int()
            n = torch.where(divisible, n // p, n)
            mask = divisible
        return r

    def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a - b
        val = _valuation(diff)
        # diff == 0 -> distance 0, else p^{-v_p}
        zero_mask = (diff == 0).all(dim=1)
        max_val = torch.max(val, dim=1).values
        dist = torch.pow(torch.tensor(float(p), device=u_c.device), -max_val.float())
        dist = torch.where(zero_mask, torch.zeros_like(dist), dist)
        return dist

    a_idx, b_idx, c_idx = idx[:, 0], idx[:, 1], idx[:, 2]
    ua, ub, uc = quant[a_idx], quant[b_idx], quant[c_idx]
    dab = _dist(ua, ub)
    dac = _dist(ua, uc)
    dbc = _dist(ub, uc)

    delta = torch.clamp(dac - torch.maximum(dab, dbc), min=0.0)
    return delta.mean()


@dataclass
class PACDTConfig:
    latent_dim_c: int = 4
    latent_dim_n: int = 4
    encoder_hidden: Sequence[int] = (128, 64)
    decoder_hidden: Sequence[int] = (64, 64)
    a_hidden: Sequence[int] = (32,)
    y_hidden: Sequence[int] = (64, 32)
    adv_hidden: Sequence[int] = (32,)
    beta_c: float = 1.0
    beta_n: float = 1.0
    lambda_a: float = 1.0
    lambda_y: float = 1.0
    gamma_adv: float = 1.0
    gamma_padic: float = 0.1
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 200
    seed: int = 42
    device: str = "cpu"
    dr_splits: int = 2


class PACDTv30Estimator(BaseCausalEstimator):
    """PACD-T v3.0 estimator with causal/nuisance split and DML ATE."""

    def __init__(self, config: Optional[PACDTConfig] = None):
        self.config = config or PACDTConfig()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.device = torch.device(self.config.device)
        self._is_fit = False

    # -------------------- public API --------------------
    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        X_all = np.asarray(X_all, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        X_all_std, self._x_mean, self._x_std = _standardize(X_all)
        Y_std, self._y_mean, self._y_std = _standardize(Y.reshape(-1, 1))
        Y_std = Y_std.squeeze(1)

        n, d_all = X_all_std.shape
        cfg = self.config
        self.encoder = _Encoder(d_all, cfg.latent_dim_c, cfg.latent_dim_n, cfg.encoder_hidden).to(self.device)
        self.proxy_decoder = _ProxyDecoder(cfg.latent_dim_c + cfg.latent_dim_n, d_all, cfg.decoder_hidden).to(self.device)
        self.a_head = _HeadA(cfg.latent_dim_c, cfg.a_hidden).to(self.device)
        self.y_head = _HeadY(cfg.latent_dim_c, cfg.y_hidden).to(self.device)
        self.adv_head = _AdversaryA(cfg.latent_dim_n, cfg.adv_hidden).to(self.device)

        optimizer_main = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.proxy_decoder.parameters())
            + list(self.a_head.parameters())
            + list(self.y_head.parameters()),
            lr=cfg.lr,
        )
        optimizer_adv = torch.optim.Adam(self.adv_head.parameters(), lr=cfg.lr)

        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_all_std), torch.from_numpy(A), torch.from_numpy(Y_std)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        for _ in range(cfg.epochs):
            for xb, ab, yb in loader:
                xb = xb.to(self.device)
                ab = ab.to(self.device)
                yb = yb.to(self.device)

                # ----- adversary update -----
                mu_c, logvar_c, mu_n, logvar_n = self.encoder(xb)
                std_n = torch.exp(0.5 * logvar_n)
                z_n = mu_n + torch.randn_like(std_n) * std_n
                logits_adv = self.adv_head(z_n.detach())
                adv_loss = bce_loss(logits_adv, ab)
                optimizer_adv.zero_grad()
                adv_loss.backward(retain_graph=True)
                optimizer_adv.step()

                # ----- main update -----
                std_c = torch.exp(0.5 * logvar_c)
                z_c = mu_c + torch.randn_like(std_c) * std_c
                z_n = mu_n + torch.randn_like(std_n) * std_n

                z_concat = torch.cat([z_c, z_n], dim=1)
                x_recon = self.proxy_decoder(z_concat)
                logits_a = self.a_head(z_c)
                y_pred = self.y_head(z_c, ab)
                logits_adv_grl = self.adv_head(_grl(z_n, cfg.gamma_adv))

                recon_loss = mse_loss(x_recon, xb)
                a_loss = bce_loss(logits_a, ab)
                y_loss = mse_loss(y_pred, yb)
                adv_loss_enc = bce_loss(logits_adv_grl, ab)

                kl_c = -0.5 * torch.mean(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
                kl_n = -0.5 * torch.mean(1 + logvar_n - mu_n.pow(2) - logvar_n.exp())
                padic_loss = _p_adic_loss(z_c)

                loss = (
                    recon_loss
                    + cfg.lambda_a * a_loss
                    + cfg.lambda_y * y_loss
                    + cfg.beta_c * kl_c
                    + cfg.beta_n * kl_n
                    + cfg.gamma_padic * padic_loss
                    + adv_loss_enc
                )

                optimizer_main.zero_grad()
                loss.backward()
                optimizer_main.step()

        self._is_fit = True

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before estimating ATE")
        U_c = self.get_latent(X_all)
        return self._dr_ate(U_c, A, Y)

    def get_latent(self, X_all: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before getting latent")
        X_all = np.asarray(X_all, dtype=np.float32)
        X_std = _apply_standardize(X_all, self._x_mean, self._x_std)
        with torch.no_grad():
            mu_c, _, _, _ = self.encoder(torch.from_numpy(X_std).to(self.device))
        return mu_c.cpu().numpy()

    # -------------------- helpers --------------------
    def _dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        kf = KFold(n_splits=self.config.dr_splits, shuffle=True, random_state=self.config.seed)
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


class PACDTree:
    """Lightweight tree partitioner used for PACD-style leaf splits.

    The tree is trained on latent features ``U`` (optionally concatenated with
    treatment as a feature) to predict outcomes and is later used purely for
    partitioning via ``apply``. This keeps compatibility with the PACD-T
    estimator while exposing a simple interface for hybrid estimators.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_leaf_size: int = 120,
        seed: int = 42,
        include_treatment: bool = True,
    ) -> None:
        self.include_treatment = include_treatment
        self._tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_leaf_size,
            random_state=seed,
        )

    def fit(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1, 1)
        Y = np.asarray(Y).reshape(-1)
        features = np.hstack([U, A]) if self.include_treatment else U
        self._tree.fit(features, Y)

    def apply(self, U: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        U = np.asarray(U)
        if self.include_treatment:
            if A is None:
                raise ValueError("Treatment vector A is required when include_treatment=True")
            features = np.hstack([U, np.asarray(A).reshape(-1, 1)])
        else:
            features = U
        return self._tree.apply(features)


__all__ = ["PACDTConfig", "PACDTv30Estimator", "PACDTree"]
