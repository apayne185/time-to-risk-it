"""Discrete-time neural hazard model (PyTorch), after Gensheimer & Narasimhan (2019).

The follow-up window is split into intervals (default: six 15-day intervals up to 90 days). An
MLP maps a player's features to one conditional hazard per interval,
h_k = P(event in interval k | no event before it). The censored-data likelihood factorises into
per-interval Bernoulli terms, so training is masked binary cross-entropy:

- an event in interval j contributes "survived" for intervals before j and "event" for j;
- a row censored at time t contributes "survived" for every interval it completed, and nothing
  for the interval it was censored in (a conservative convention; split-boundary censoring at
  day 30 falls exactly on an interval edge).

P(event within h days) = 1 - prod(1 - h_k) over intervals up to h, with log-linear interpolation
inside an interval. The 30-day decision horizon is an interval edge, so it needs none.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ttr.models.base import Preprocessor, SurvivalData, SurvivalModel

DEFAULT_EDGES = (0, 15, 30, 45, 60, 75, 90)


def interval_targets(
    time: np.ndarray, event: np.ndarray, edges: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval targets ``y`` (1 = event in that interval) and ``mask`` (1 = observed)."""
    lower, upper = np.asarray(edges[:-1]), np.asarray(edges[1:])
    t = time[:, None]
    in_interval = (t > lower) & (t <= upper)
    completed = t >= upper
    is_event = event[:, None] == 1
    y = (in_interval & is_event).astype(np.float32)
    mask = (completed | (in_interval & is_event)).astype(np.float32)
    # An event beyond the last edge is censored at the horizon.
    return y, mask


class _HazardMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layers(x)
        return out  # logits of the per-interval hazards


def masked_nll(
    logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Weighted negative log-likelihood per row of the discrete-time survival model."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    per_row = (bce * mask).sum(dim=1)
    return (per_row * weight).sum() / weight.sum()


class DiscreteTimeHazardNet(SurvivalModel):
    name = "torch_hazard"

    def __init__(
        self,
        hidden: int = 64,
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        lr: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 200,
        patience: int = 15,
        edges: tuple[int, ...] = DEFAULT_EDGES,
        seed: int = 0,
        num_boost_round: int | None = None,
    ) -> None:
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        # Refit path: train for a fixed number of epochs found by early stopping (the shared
        # training code passes it as num_boost_round, as for XGBoost).
        self.max_epochs = num_boost_round or max_epochs
        self.fixed_epochs = num_boost_round is not None
        self.patience = patience
        self.edges = tuple(edges)
        self.seed = seed

    # -- data -------------------------------------------------------------------------------
    def _tensors(self, data: SurvivalData) -> tuple[torch.Tensor, ...]:
        y, mask = interval_targets(data.time, data.event, self.edges)
        w = np.ones(len(data)) if data.weight is None else data.weight
        return (
            self._inputs(data.X),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            torch.as_tensor(w, dtype=torch.float32),
        )

    def _inputs(self, X: pd.DataFrame) -> torch.Tensor:
        Z = self.pre_.transform(X)[self.columns_]
        return torch.as_tensor(Z.to_numpy(dtype=np.float32))

    # -- training ---------------------------------------------------------------------------
    def fit(self, data: SurvivalData, valid: SurvivalData | None = None) -> DiscreteTimeHazardNet:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.pre_ = Preprocessor().fit(data.X)
        self.columns_ = list(self.pre_.transform(data.X.head(1)).columns)
        self.net_ = _HazardMLP(len(self.columns_), len(self.edges) - 1, self.hidden, self.dropout)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        X, y, mask, w = self._tensors(data)
        val = self._tensors(valid) if valid is not None and not self.fixed_epochs else None

        best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0
        self.history_: list[float] = []
        for epoch in range(self.max_epochs):
            self.net_.train()
            for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // self.batch_size)):
                b = torch.from_numpy(idx)
                opt.zero_grad()
                loss = masked_nll(self.net_(X[b]), y[b], mask[b], w[b])
                loss.backward()  # type: ignore[no-untyped-call]  # torch stubs leave it untyped
                opt.step()
            if val is None:
                continue
            self.net_.eval()
            with torch.no_grad():
                v = float(masked_nll(self.net_(val[0]), val[1], val[2], val[3]))
            self.history_.append(v)
            if v < best_loss - 1e-5:
                best_loss, best_epoch, waited = v, epoch, 0
                best_state = {k: t.clone() for k, t in self.net_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.best_iteration_ = best_epoch if val is not None else self.max_epochs - 1
        self.net_.eval()
        return self

    # -- prediction -------------------------------------------------------------------------
    def hazards(self, X: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            out: np.ndarray = torch.sigmoid(self.net_(self._inputs(X))).numpy().astype(float)
        return out

    def predict_event_prob(self, X: pd.DataFrame, horizon: float) -> np.ndarray:
        log_surv = np.log1p(-np.clip(self.hazards(X), 0, 1 - 1e-7))
        cum = np.concatenate([np.zeros((len(X), 1)), np.cumsum(log_surv, axis=1)], axis=1)
        edges = np.asarray(self.edges, dtype=float)
        h = float(np.clip(horizon, edges[0], edges[-1]))
        k = min(int(np.searchsorted(edges, h, side="right")) - 1, len(edges) - 2)
        frac = (h - edges[k]) / (edges[k + 1] - edges[k])
        out: np.ndarray = 1 - np.exp(cum[:, k] + frac * (cum[:, k + 1] - cum[:, k]))
        return out

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Cumulative hazard over the full horizon (monotone in the event probability)."""
        out: np.ndarray = -np.log1p(
            -np.clip(self.predict_event_prob(X, self.edges[-1]), 0, 1 - 1e-12)
        )
        return out

    def params(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "lr": self.lr,
            "edges": ",".join(map(str, self.edges)),
            "epochs": getattr(self, "best_iteration_", -1) + 1,
        }
