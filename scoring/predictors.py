"""
Loads the 3 trained models + their preprocessors and exposes a simple
predict interface. Both score_book.py (batch scoring of the full book)
and app.py (single live prediction in the Interface tab) import this.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from feature_config import LOSS_FEATURES, BIND_FEATURES  # noqa: E402
from preprocessing import engineer_bind_features  # noqa: E402

MODEL_DIR = Path(__file__).parent.parent / "models"


class ModelBundle:
    def __init__(self):
        self.freq_model = joblib.load(MODEL_DIR / "frequency_model.pkl")
        self.freq_prep = joblib.load(MODEL_DIR / "frequency_preprocessor.pkl")
        self.sev_model = joblib.load(MODEL_DIR / "severity_model.pkl")
        self.sev_prep = joblib.load(MODEL_DIR / "severity_preprocessor.pkl")
        self.bind_model = joblib.load(MODEL_DIR / "bind_model.pkl")
        self.bind_prep = joblib.load(MODEL_DIR / "bind_preprocessor.pkl")

    def predict_frequency(self, df: pd.DataFrame) -> pd.Series:
        X = self.freq_prep.transform(df[LOSS_FEATURES])
        return pd.Series(self.freq_model.predict(X), index=df.index).clip(lower=0.005, upper=1.5)

    def predict_severity(self, df: pd.DataFrame) -> pd.Series:
        X = self.sev_prep.transform(df[LOSS_FEATURES])
        X_sm = sm.add_constant(X, has_constant="add")
        preds = self.sev_model.predict(X_sm)
        return pd.Series(preds, index=df.index).clip(lower=500, upper=100_000)

    def predict_bind_probability(self, df: pd.DataFrame) -> pd.Series:
        df = engineer_bind_features(df)
        X = self.bind_prep.transform(df[BIND_FEATURES])
        proba = self.bind_model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=df.index)


_bundle: ModelBundle | None = None


def get_model_bundle() -> ModelBundle:
    """Lazily load models once and reuse (Streamlit wraps this with
    st.cache_resource in app.py)."""
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle
