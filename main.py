# main.py

from src.load_data import load_f1_data
from src.prepare_features import prepare_features
from src.train_model import train_model, TrainConfig
from src.evaluate import evaluate_model


def main():
    # 1) Load data
    constructors, constructor_standings, races, results = load_f1_data()

    # 2) Build features/labels
    X, y, years, constructor_ids = prepare_features(constructors, constructor_standings, races)

    # 3) Time-based train/test split (avoid leakage)
    train_mask = years <= 2015
    test_mask = years > 2015

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    years_test = years[test_mask]
    constructor_ids_test = constructor_ids[test_mask]

    # 4) Train model
    config = TrainConfig(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model = train_model(X_train, y_train, config=config)

    # 5) Evaluate model (includes Top-1 per-year pick accuracy if predict_proba exists)
    _pred_df = evaluate_model(
        model,
        X_test,
        y_test,
        years=years_test,
        constructor_ids=constructor_ids_test,
    )

    # Optional: print a few highest-probability predictions (across all test rows)
    print("\n=== Highest predicted champion probabilities (sample) ===")
    print(_pred_df.sort_values("y_proba", ascending=False).head(10))


if __name__ == "__main__":
    main()
