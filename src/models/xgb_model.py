"""XGBoost baseline for triple-barrier classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
}

_LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
_CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}


def _map_labels(y) -> np.ndarray:
    y = np.asarray(y).astype(np.int64)
    out = np.empty_like(y)
    for raw, cls in _LABEL_TO_CLASS.items():
        out[y == raw] = cls
    return out


def train_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
) -> xgb.Booster:
    """Train a multiclass XGBoost model with early stopping on val."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    dtrain = xgb.DMatrix(X_train, label=_map_labels(y_train))
    dval = xgb.DMatrix(X_val, label=_map_labels(y_val))

    booster = xgb.train(
        params=p,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster


def predict_xgb(booster: xgb.Booster, X) -> np.ndarray:
    """Return class probabilities with shape [N, 3] for classes {0,1,2} = {-1,0,+1}."""
    dmat = xgb.DMatrix(X)
    kwargs: dict[str, Any] = {}
    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
        kwargs["iteration_range"] = (0, booster.best_iteration + 1)
    probs = booster.predict(dmat, **kwargs)
    return np.asarray(probs).reshape(-1, 3)


def predict_xgb_signed(booster: xgb.Booster, X) -> np.ndarray:
    """Return expected sign p(+1) - p(-1) in [-1, 1]."""
    probs = predict_xgb(booster, X)
    # class 0 = -1 (down), class 2 = +1 (up)
    return probs[:, 2] - probs[:, 0]