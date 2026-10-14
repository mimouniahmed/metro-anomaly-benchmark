"""Autoencoders trained to reconstruct normal behaviour; the reconstruction error is the anomaly score.

Two models:

- a dense autoencoder on the 18 window features (one row per minute);
- an LSTM autoencoder on raw signal sequences (2 hours of 5-minute means, several signals).

Both are trained on the training period only, with the last 10% of it held out for early
stopping, and score each minute causally (a sequence is scored at its last time step).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn

SIGNALS = ["Motor_current", "DV_eletric", "Oil_temperature", "TP2", "Reservoirs"]


class DenseAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 12, code: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, code), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTMAutoencoder(nn.Module):
    """Encodes a sequence into the last hidden state, then decodes the whole sequence from it."""

    def __init__(self, n_signals: int, hidden: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(n_signals, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_signals)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        repeated = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def _train(model: nn.Module, data: np.ndarray, seed: int, epochs: int = 100, batch_size: int = 256,
           lr: float = 1e-3, patience: int = 5) -> tuple[nn.Module, list[tuple[float, float]]]:
    """Adam + MSE with early stopping on the last 10% of the (time-ordered) training data."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    split = int(len(data) * 0.9)
    fit, val = torch.tensor(data[:split], dtype=torch.float32), torch.tensor(data[split:], dtype=torch.float32)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, wait, history = np.inf, None, 0, []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(fit))
        total = 0.0
        for start in range(0, len(fit), batch_size):
            batch = fit[order[start:start + batch_size]]
            optimiser.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val), val).item()
        history.append((total / len(fit), val_loss))
        if val_loss < best - 1e-5:
            best, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def _errors(model: nn.Module, data: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Squared reconstruction error per input element (same shape as ``data``)."""
    out = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = torch.tensor(data[start:start + batch_size], dtype=torch.float32)
            out.append(((model(batch) - batch) ** 2).numpy())
    return np.concatenate(out) if out else np.empty((0,) + data.shape[1:])


def dense_autoencoder(X: pd.DataFrame, train: pd.DataFrame, seed: int = 0) -> tuple[pd.Series, pd.DataFrame, list]:
    """Mean squared reconstruction error of the window features, its per-feature breakdown, and the loss history."""
    train = train.dropna()
    scaler = RobustScaler().fit(train)
    complete = X.dropna()
    model, history = _train(DenseAutoencoder(X.shape[1]), scaler.transform(train), seed)
    errors = pd.DataFrame(_errors(model, scaler.transform(complete)), index=complete.index, columns=X.columns)
    return errors.mean(axis=1).reindex(X.index), errors.reindex(X.index), history


def make_sequences(grid: pd.DataFrame, signals=SIGNALS, length: int = 24, step: str = "5min"):
    """All complete sequences of ``length`` steps of ``step`` means, indexed by their last time step."""
    coarse = grid[signals].resample(step).mean()
    values = coarse.to_numpy(dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(values, (length, len(signals)))[:, 0]
    ends = coarse.index[length - 1:]
    complete = ~np.isnan(windows).any(axis=(1, 2))
    return windows[complete], ends[complete], coarse


@dataclass
class LSTMFit:
    """A trained LSTM autoencoder with the scaling it was trained with."""

    model: LSTMAutoencoder
    scaler: RobustScaler
    signals: list
    length: int
    step: str

    def reconstruct(self, grid: pd.DataFrame, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The sequence ending at ``end`` and its reconstruction, both in original units."""
        coarse = grid[self.signals].resample(self.step).mean().loc[:end].iloc[-self.length:]
        scaled = self.scaler.transform(coarse).astype(np.float32)
        with torch.no_grad():
            rebuilt = self.model(torch.tensor(scaled[None])).numpy()[0]
        return coarse, pd.DataFrame(self.scaler.inverse_transform(rebuilt), index=coarse.index, columns=self.signals)


def lstm_autoencoder(grid: pd.DataFrame, train_end: pd.Timestamp, exclude: pd.DataFrame | None = None,
                     signals=SIGNALS, length: int = 24, step: str = "5min",
                     seed: int = 0) -> tuple[pd.Series, list, LSTMFit]:
    """Mean squared reconstruction error of each 2-hour sequence, held on the minute grid.

    Returns ``(score, loss history, fitted model)``.

    Signals are robust-scaled with training statistics. Training sequences overlapping an
    ``exclude`` interval (leak-like episodes found by the expert rule) are left out of the fit.
    """
    coarse_train = grid.loc[grid.index < train_end, signals].resample(step).mean()
    scaler = RobustScaler().fit(coarse_train.dropna())
    scaled = pd.DataFrame(scaler.transform(grid[signals]), index=grid.index, columns=signals)
    windows, ends, _ = make_sequences(scaled, signals, length, step)
    span = (length - 1) * pd.Timedelta(step)
    is_train = ends < train_end
    keep = is_train.copy()
    for start, end in ([] if exclude is None else exclude[["start", "end"]].itertuples(index=False)):
        keep &= ~((ends >= start) & (ends - span <= end))
    model, history = _train(LSTMAutoencoder(len(signals)), windows[keep], seed, batch_size=128)
    score = pd.Series(_errors(model, windows).mean(axis=(1, 2)), index=ends)
    minute = score.reindex(grid.index, method="ffill", limit=int(pd.Timedelta(step) / pd.Timedelta("1min")) - 1)
    return minute.where(grid["valid"]), history, LSTMFit(model, scaler, list(signals), length, step)
