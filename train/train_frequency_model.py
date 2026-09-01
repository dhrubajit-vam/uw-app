"""
Train the CLAIM FREQUENCY model.

Model:    XGBoost Regressor, objective="count:poisson"
Target:   Claim_Count  (number of claims filed)
Trained:  only on rows where Bound_Flag == 1 (you only observe claims on
          policies that were actually written)
Exposure: every row = 1 policy-year, so Claim_Count IS already a rate;
          no separate offset column is needed for this synthetic annual data.

Run:
    python train/train_frequency_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from feature_config import LOSS_FEATURES, LOSS_NUMERIC_FEATURES, LOSS_CATEGORICAL_FEATURES
from preprocessing import build_preprocessor
from splits import time_based_split

DATA_PATH = Path(__file__).parent.parent / "data" / "scored_dataset_full.parquet"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_parquet(DATA_PATH)
    df = df[df["Bound_Flag"] == 1].copy()  # can only learn frequency on written business

    train, test, cutoff = time_based_split(df)
    print(f"Frequency model | train={len(train):,} test={len(test):,} (split at {cutoff.date()})")

    preprocessor = build_preprocessor(LOSS_NUMERIC_FEATURES, LOSS_CATEGORICAL_FEATURES)
    X_train = preprocessor.fit_transform(train[LOSS_FEATURES])
    X_test = preprocessor.transform(test[LOSS_FEATURES])
    y_train = train["Claim_Count"].values
    y_test = test["Claim_Count"].values

    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_DIR / "frequency_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "frequency_preprocessor.pkl")

    # persist test set predictions for the console evaluation script
    test_out = test[["Policy_ID", "Claim_Count"]].copy()
    test_out["Predicted_Frequency"] = model.predict(X_test)
    test_out.to_parquet(MODEL_DIR / "frequency_test_predictions.parquet", index=False)

    print("Saved: frequency_model.pkl, frequency_preprocessor.pkl")
    print(f"Mean predicted frequency (test): {test_out['Predicted_Frequency'].mean():.4f}")
    print(f"Mean actual frequency (test):    {y_test.mean():.4f}")


if __name__ == "__main__":
    main()
