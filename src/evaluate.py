from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    years: Optional[pd.Series] = None,
    constructor_ids: Optional[pd.Series] = None,
    show_reports: bool = True,
) -> pd.DataFrame:

    # --- Predictions ---
    y_pred = model.predict(X_test)

    # Probability of class 1 (champion)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback if model doesn't support predict_proba
        y_proba = np.full(shape=len(X_test), fill_value=np.nan)

    # --- Build result dataframe ---
    pred_df = pd.DataFrame({
        "y_true": pd.to_numeric(y_test, errors="coerce").astype(int).values,
        "y_pred": pd.to_numeric(y_pred, errors="coerce").astype(int),
        "y_proba": y_proba,
    })

    if years is not None:
        pred_df["year"] = years.values
    if constructor_ids is not None:
        pred_df["constructor_id"] = constructor_ids.values

    # --- Standard classification metrics ---
    if show_reports:
        print("\n=== Standard Metrics ===")
        print("Accuracy:", accuracy_score(pred_df["y_true"], pred_df["y_pred"]))
        print("\nConfusion Matrix:")
        print(confusion_matrix(pred_df["y_true"], pred_df["y_pred"]))
        print("\nClassification Report:")
        print(classification_report(pred_df["y_true"], pred_df["y_pred"], digits=4))

    # --- Top-1 Champion per Year evaluation (recommended) ---
    # This measures whether the model picks the correct champion team per season.
    if years is not None and pred_df["y_proba"].notna().all():
        # For each year, choose row with max predicted champion probability
        top1 = pred_df.sort_values(["year", "y_proba"], ascending=[True, False]) \
                     .groupby("year", as_index=False) \
                     .head(1)

        top1_acc = (top1["y_true"] == 1).mean()  # True champion should be 1 for correct pick

        if show_reports:
            print("\n=== Top-1 Champion Pick Accuracy (per year) ===")
            print(f"Top-1 per-year accuracy: {top1_acc:.4f}")
            print("\nSample of yearly picks (first 10):")
            print(top1.head(10))

    elif years is not None and show_reports:
        print("\n(Note) 'years' provided but model has no predict_proba(), so Top-1 per year accuracy was skipped.")

    return pred_df
