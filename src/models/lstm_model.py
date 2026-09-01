"""PyTorch LSTM for triple-barrier classification on feature sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.features import FEATURES

# Last L bars ending at t → one sample; L=50 = 5 s at 100 ms bars.
SEQ_LEN = 50
N_FEATURES = len(FEATURES)  # 12
N_CLASSES = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Same class map as XGBoost: CrossEntropyLoss wants {0, 1, 2}.
_LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}

# Avoid divide-by-zero on constant columns.
_STD_EPS = 1e-8


class FeatureScaler:
    """Per-column (x - mean) / std. Fit on train only — never on val/test."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> FeatureScaler:
        """Compute mean/std over finite rows. ``X`` shape ``[N, F]``."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D [N, F], got shape {X.shape}")
        # nanmean/nanstd: a stray NaN must not poison an entire column.
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std = np.where(~np.isfinite(std) | (std < _STD_EPS), 1.0, std)
        mean = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return float32 scaled array; same shape as ``X``."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler must be fit before transform")
        X = np.asarray(X, dtype=np.float64)
        out = (X - self.mean_) / self.std_
        return out.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler must be fit before save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str | Path) -> FeatureScaler:
        data = np.load(path)
        scaler = cls()
        scaler.mean_ = np.asarray(data["mean"], dtype=np.float64)
        scaler.std_ = np.asarray(data["std"], dtype=np.float64)
        return scaler


class SequenceDataset(Dataset):
    """Sliding windows of features with the label at the window's last bar.

    Sample i ending at row ``t`` is:
      X = features[t - L + 1 : t + 1]   shape [L, F]
      y = label[t]                      class in {0, 1, 2}
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list[str] | None = None,
        label_col: str = "label_tb",
        seq_len: int = SEQ_LEN,
    ) -> None:
        feature_cols = feature_cols or FEATURES
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        missing = [c for c in feature_cols + [label_col] if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        self.seq_len = seq_len
        self.feature_cols = list(feature_cols)
        self.label_col = label_col

        # Materialize once so __getitem__ is cheap (numpy slices, not Polars).
        self.X = df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
        raw_y = df.select(label_col).to_numpy().reshape(-1).astype(np.int64, copy=False)
        self.y = np.empty_like(raw_y)
        for raw, cls in _LABEL_TO_CLASS.items():
            self.y[raw_y == raw] = cls

        self.valid_indices = self._build_valid_indices()

    def _build_valid_indices(self) -> np.ndarray:
        """Row ends ``t`` where the full window is in-bounds and finite."""
        n = self.X.shape[0]
        L = self.seq_len
        if n < L:
            return np.array([], dtype=np.int64)

        # Vectorized: window [t-L+1, t] is finite iff no bad rows in that span.
        finite_row = np.isfinite(self.X).all(axis=1)
        bad = (~finite_row).astype(np.int32)
        csum = np.concatenate([[0], np.cumsum(bad)])
        t = np.arange(L - 1, n, dtype=np.int64)
        bad_in_window = csum[t + 1] - csum[t - L + 1]
        label_ok = (self.y[t] >= 0) & (self.y[t] <= 2)
        return t[(bad_in_window == 0) & label_ok]

    def __len__(self) -> int:
        return int(self.valid_indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(features[L, F], label)`` as float32 / int64 tensors."""
        t = int(self.valid_indices[idx])
        L = self.seq_len
        x = self.X[t - L + 1 : t + 1]  # [L, F]
        y = self.y[t]
        return (
            torch.from_numpy(x.copy()),  # copy: numpy slice may not own memory
            torch.tensor(y, dtype=torch.int64),
        )


class MNQLSTM(nn.Module):
    """2-layer LSTM → linear head for {-1, 0, +1} as classes {0, 1, 2}.

    Input ``x`` has shape ``[batch, seq_len, n_features]`` (``batch_first=True``).
    We classify from the hidden state at the **last** timestep only.
    """

    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: int = N_CLASSES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``[batch, num_classes]``."""
        # out: [batch, seq_len, hidden_size]
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # hidden state after seeing the full window
        return self.head(last)


def _epoch_loss_acc(
    model: MNQLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over ``loader``. If ``optimizer`` is set, run a train step."""
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += bs

    if total_n == 0:
        return float("nan"), float("nan")
    return total_loss / total_n, total_correct / total_n


def train_lstm(
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    num_workers: int = 2,
    patience: int = 3,
    device: str | torch.device | None = None,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = NUM_LAYERS,
) -> MNQLSTM:
    """Train ``MNQLSTM`` with Adam + CrossEntropy; early-stop on val loss.

    Logs per-epoch ``train/loss``, ``train/acc``, ``val/loss``, ``val/acc`` to W&B
    when a run is active. Returns the model with best-val-loss weights restored.
    """
    import wandb

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MNQLSTM(
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _epoch_loss_acc(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        with torch.no_grad():
            val_loss, val_acc = _epoch_loss_acc(
                model, val_loader, criterion, device, optimizer=None
            )

        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
        }
        print(
            f"epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if wandb.run is not None:
            wandb.log(metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"early stop at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model

