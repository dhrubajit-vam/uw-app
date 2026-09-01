"""
Train the CLAIM SEVERITY model.

Model:    Gamma GLM (statsmodels), log link
Target:   Incurred_Loss / Claim_Count  (average $ cost PER claim)
Trained:  only on rows where Claim_Count > 0 - you can't have a "cost per
          claim" for a policy that had zero claims.

Run:
    python train/train_severity_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from feature_config import LOSS_FEATURES, LOSS_NUMERIC_FEATURES, LOSS_CATEGORICAL_FEATURES
from preprocessing import build_preprocessor
from splits import time_based_split

DATA_PATH = Path(__file__).parent.parent / "data" / "scored_dataset_full.parquet"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_parquet(DATA_PATH)
    df = df[(df["Bound_Flag"] == 1) & (df["Claim_Count"] > 0)].copy()
    df["Severity"] = df["Incurred_Loss"] / df["Claim_Count"]
    # Gamma GLM requires strictly positive target
    df = df[df["Severity"] > 0].copy()

    train, test, cutoff = time_based_split(df)
    print(f"Severity model | train={len(train):,} test={len(test):,} (split at {cutoff.date()})")

    preprocessor = build_preprocessor(LOSS_NUMERIC_FEATURES, LOSS_CATEGORICAL_FEATURES)
    X_train = preprocessor.fit_transform(train[LOSS_FEATURES])
    X_test = preprocessor.transform(test[LOSS_FEATURES])

    # statsmodels GLM wants a constant term
    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm = sm.add_constant(X_test, has_constant="add")

    y_train = train["Severity"].values
    y_test = test["Severity"].values

    glm = sm.GLM(y_train, X_train_sm, family=sm.families.Gamma(link=sm.families.links.Log()))
    result = glm.fit()

    joblib.dump(result, MODEL_DIR / "severity_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "severity_preprocessor.pkl")

    preds = result.predict(X_test_sm)
    test_out = test[["Policy_ID"]].copy()
    test_out["Actual_Severity"] = y_test
    test_out["Predicted_Severity"] = preds
    test_out.to_parquet(MODEL_DIR / "severity_test_predictions.parquet", index=False)

    print("Saved: severity_model.pkl, severity_preprocessor.pkl")
    print(f"Mean predicted severity (test): ${preds.mean():,.0f}")
    print(f"Mean actual severity (test):    ${y_test.mean():,.0f}")
    print(f"GLM deviance: {result.deviance:.2f}  |  AIC: {result.aic:.2f}")


if __name__ == "__main__":
    main()
