"""IVAPCI v2.1 implementation following docs/IVAPCI_v2.1_spec.md."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from . import BaseCausalEstimator


def _standardize(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, mean, std


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
        last = input_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(last, latent_dim)
        self.logvar = nn.Linear(last, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


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
        last = latent_dim + 1  # concat A
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a.unsqueeze(-1)], dim=1)).squeeze(-1)


@dataclass
class IVAPCIConfig:
    latent_dim: int = 4
    encoder_hidden: Sequence[int] = (128, 64)
    decoder_hidden: Sequence[int] = (64, 64)
    a_hidden: Sequence[int] = (32,)
    y_hidden: Sequence[int] = (64, 32)
    beta: float = 1.0
    lambda_a: float = 1.0
    lambda_y: float = 1.0
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 200
    seed: int = 42
    device: str = "cpu"


class IVAPCIv21Estimator(BaseCausalEstimator):
    """IVAPCI v2.1 estimator implementing VAE+AY supervision and DR ATE."""

    def __init__(self, config: Optional[IVAPCIConfig] = None):
        self.config = config or IVAPCIConfig()
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
        self.encoder = _Encoder(d_all, cfg.latent_dim, cfg.encoder_hidden).to(self.device)
        self.proxy_decoder = _ProxyDecoder(cfg.latent_dim, d_all, cfg.decoder_hidden).to(self.device)
        self.a_head = _HeadA(cfg.latent_dim, cfg.a_hidden).to(self.device)
        self.y_head = _HeadY(cfg.latent_dim, cfg.y_hidden).to(self.device)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.proxy_decoder.parameters())
            + list(self.a_head.parameters())
            + list(self.y_head.parameters()),
            lr=cfg.lr,
        )

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

                mu, logvar = self.encoder(xb)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

                x_recon = self.proxy_decoder(z)
                logits_a = self.a_head(z)
                y_pred = self.y_head(z, ab)

                recon_loss = mse_loss(x_recon, xb)
                a_loss = bce_loss(logits_a, ab)
                y_loss = mse_loss(y_pred, yb)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                loss = recon_loss + cfg.lambda_a * a_loss + cfg.lambda_y * y_loss + cfg.beta * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._is_fit = True

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before estimating ATE")
        U = self.get_latent(X_all)
        return self._dr_ate(U, A, Y)

    def get_latent(self, X_all: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before getting latent")
        X_all = np.asarray(X_all, dtype=np.float32)
        X_std = _apply_standardize(X_all, self._x_mean, self._x_std)
        with torch.no_grad():
            mu, _ = self.encoder(torch.from_numpy(X_std).to(self.device))
        return mu.cpu().numpy()

    # -------------------- helpers --------------------
    def _dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        kf = KFold(n_splits=2, shuffle=True, random_state=self.config.seed)
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


__all__ = ["IVAPCIConfig", "IVAPCIv21Estimator"]
