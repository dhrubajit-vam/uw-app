"""
Shared preprocessing helpers.

We keep this deliberately simple and consistent across all 3 models:
  - numeric features: median-imputed
  - categorical features: most-frequent-imputed, then one-hot encoded
  - the fitted ColumnTransformer is pickled alongside each model so the
    Streamlit app applies IDENTICAL preprocessing at inference time.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def engineer_bind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns needed by the bind model (called both at
    training time and at inference time so the two stay identical)."""
    df = df.copy()
    df["Premium_Change_Percentage_Input"] = (
        df["Final_Quoted_Premium"] / df["Prior_Year_Premium"].replace(0, pd.NA) - 1
    ).fillna(0.0)
    return df


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())
