from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


@dataclass
class TrainConfig:
    """
    Simple config so you can tweak the model without hunting through code.
    """
    n_estimators: int = 300
    max_depth: Optional[int] = 6
    random_state: int = 42
    class_weight: str | dict | None = "balanced"  # helps with imbalanced classes
    n_jobs: int = -1  # use all CPU cores


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainConfig = TrainConfig()
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the provided features/labels.

    Args:
        X_train: DataFrame of features
        y_train: Series of labels (0/1)
        config: TrainConfig with hyperparameters

    Returns:
        Trained RandomForestClassifier
    """
    # Basic safety checks (helpful when debugging)
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train cannot be None.")

    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train must have same length. Got {len(X_train)} and {len(y_train)}.")

    if X_train.isna().any().any():
        raise ValueError("X_train contains NaN values. Check feature engineering (lag features can create NaNs).")

    # Ensure y is integer 0/1
    y_train_clean = pd.to_numeric(y_train, errors="coerce").astype(int)

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        class_weight=config.class_weight,
        n_jobs=config.n_jobs,
    )

    model.fit(X_train, y_train_clean)
    return model