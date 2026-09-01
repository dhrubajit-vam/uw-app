"""
Train the BIND PROPENSITY model.

Model:    LightGBM Classifier
Target:   Bound_Flag (did the customer accept the quote?)
Trained:  on the full book (every quoted submission has a bind outcome,
          bound or not - unlike claims, this isn't conditional on binding).

Run:
    python train/train_bind_model.py
"""
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from feature_config import BIND_FEATURES, BIND_NUMERIC_FEATURES, BIND_CATEGORICAL_FEATURES
from preprocessing import build_preprocessor, engineer_bind_features
from splits import time_based_split

DATA_PATH = Path(__file__).parent.parent / "data" / "scored_dataset_full.parquet"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_parquet(DATA_PATH)
    df = engineer_bind_features(df)

    train, test, cutoff = time_based_split(df)
    print(f"Bind model | train={len(train):,} test={len(test):,} (split at {cutoff.date()})")

    preprocessor = build_preprocessor(BIND_NUMERIC_FEATURES, BIND_CATEGORICAL_FEATURES)
    X_train = preprocessor.fit_transform(train[BIND_FEATURES])
    X_test = preprocessor.transform(test[BIND_FEATURES])
    y_train = train["Bound_Flag"].values
    y_test = test["Bound_Flag"].values

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_DIR / "bind_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "bind_preprocessor.pkl")

    proba = model.predict_proba(X_test)[:, 1]
    test_out = test[["Policy_ID", "Bound_Flag"]].copy()
    test_out["Predicted_Bind_Probability"] = proba
    test_out.to_parquet(MODEL_DIR / "bind_test_predictions.parquet", index=False)

    print("Saved: bind_model.pkl, bind_preprocessor.pkl")
    print(f"Mean predicted bind prob (test): {proba.mean():.4f}")
    print(f"Actual bind rate (test):         {y_test.mean():.4f}")


if __name__ == "__main__":
    main()
