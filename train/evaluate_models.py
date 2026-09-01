"""
CONSOLE-ONLY model performance report. This is intentionally NOT part of
the Streamlit app - it's the equivalent of the workbook's QA_Findings tab,
run by a developer/actuary after training to sanity-check the models
before trusting them in the app.

Run:
    python train/evaluate_models.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

MODEL_DIR = Path(__file__).parent.parent / "models"

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def line(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def lift_table(df: pd.DataFrame, actual_col: str, predicted_col: str, n_bins=10, money=False):
    """The single most important actuarial check: bucket by predicted risk
    decile and confirm actual outcome increases monotonically."""
    d = df.copy()
    d["decile"] = pd.qcut(d[predicted_col].rank(method="first"), n_bins, labels=False) + 1
    summary = d.groupby("decile").agg(
        n=(predicted_col, "size"),
        avg_predicted=(predicted_col, "mean"),
        avg_actual=(actual_col, "mean"),
    )
    fmt = "${:,.0f}" if money else "{:.4f}"
    summary["avg_predicted"] = summary["avg_predicted"].apply(lambda x: fmt.format(x))
    summary["avg_actual"] = summary["avg_actual"].apply(lambda x: fmt.format(x))
    print(summary.to_string())
    return summary


# ---------------------------------------------------------------------------
# FREQUENCY MODEL
# ---------------------------------------------------------------------------
line("CLAIM FREQUENCY MODEL  (XGBoost, Poisson objective)")
freq = pd.read_parquet(MODEL_DIR / "frequency_test_predictions.parquet")
mae = mean_absolute_error(freq["Claim_Count"], freq["Predicted_Frequency"])
# Poisson deviance requires non-negative predictions
pred_clip = freq["Predicted_Frequency"].clip(lower=1e-6)
dev = mean_poisson_deviance(freq["Claim_Count"], pred_clip)
print(f"Test rows:              {len(freq):,}")
print(f"MAE:                    {mae:.4f}")
print(f"Mean Poisson Deviance:  {dev:.4f}   (lower is better)")
print(f"Actual mean frequency:  {freq['Claim_Count'].mean():.4f}")
print(f"Predicted mean freq:    {freq['Predicted_Frequency'].mean():.4f}")
print("\nLift table (decile 1 = lowest predicted risk -> 10 = highest).")
print("HEALTHY MODEL: avg_actual should rise monotonically (or close to it) across deciles.\n")
lift_table(freq, "Claim_Count", "Predicted_Frequency")

# ---------------------------------------------------------------------------
# SEVERITY MODEL
# ---------------------------------------------------------------------------
line("CLAIM SEVERITY MODEL  (Gamma GLM)")
sev = pd.read_parquet(MODEL_DIR / "severity_test_predictions.parquet")
mae_sev = mean_absolute_error(sev["Actual_Severity"], sev["Predicted_Severity"])
mape = (
    (sev["Actual_Severity"] - sev["Predicted_Severity"]).abs() / sev["Actual_Severity"]
).mean()
print(f"Test rows (claims only): {len(sev):,}")
print(f"MAE:                     ${mae_sev:,.0f}")
print(f"MAPE:                    {mape:.1%}")
print(f"Actual mean severity:    ${sev['Actual_Severity'].mean():,.0f}")
print(f"Predicted mean severity: ${sev['Predicted_Severity'].mean():,.0f}")
print("\nLift table (decile 1 = lowest predicted severity -> 10 = highest).\n")
lift_table(sev, "Actual_Severity", "Predicted_Severity", n_bins=5, money=True)  # small n -> 5 bins

# ---------------------------------------------------------------------------
# BIND MODEL
# ---------------------------------------------------------------------------
line("BIND PROPENSITY MODEL  (LightGBM Classifier)")
bind = pd.read_parquet(MODEL_DIR / "bind_test_predictions.parquet")
auc = roc_auc_score(bind["Bound_Flag"], bind["Predicted_Bind_Probability"])
ll = log_loss(bind["Bound_Flag"], bind["Predicted_Bind_Probability"])
brier = brier_score_loss(bind["Bound_Flag"], bind["Predicted_Bind_Probability"])
print(f"Test rows:        {len(bind):,}")
print(f"AUC:              {auc:.4f}   (0.5 = random, 1.0 = perfect)")
print(f"Log loss:         {ll:.4f}   (lower is better)")
print(f"Brier score:      {brier:.4f}   (lower is better)")
print(f"Actual bind rate: {bind['Bound_Flag'].mean():.4f}")

print("\nCalibration check (predicted decile vs actual bind rate in that bucket).")
print("HEALTHY MODEL: predicted and actual should track closely across deciles.\n")
cal = bind.copy()
cal["decile"] = pd.qcut(cal["Predicted_Bind_Probability"].rank(method="first"), 10, labels=False) + 1
cal_summary = cal.groupby("decile").agg(
    n=("Bound_Flag", "size"),
    avg_predicted_prob=("Predicted_Bind_Probability", "mean"),
    actual_bind_rate=("Bound_Flag", "mean"),
)
print(cal_summary.to_string())

line("DONE - review lift/calibration tables above before trusting the app's predictions.")
