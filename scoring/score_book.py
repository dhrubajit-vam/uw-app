"""
Scores the ENTIRE historical book (22,000 rows) using the trained models
+ deterministic rules engine. Output feeds the Streamlit Dashboard tab -
this replaces what used to be static formula columns in the Excel
Scored_Dataset_Full sheet.

Run (after all 3 train_*.py scripts have been run):
    python scoring/score_book.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "rules"))
from predictors import get_model_bundle  # noqa: E402
from row_mapper import row_to_submission_inputs  # noqa: E402
from rules_engine import score_submission  # noqa: E402

DATA_PATH = Path(__file__).parent.parent / "data" / "scored_dataset_full.parquet"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "model_scored_dataset.parquet"
MARKET_ADJ_PATH = MODEL_DIR / "market_adj_by_state.json"


def build_market_adj_lookup(df: pd.DataFrame) -> dict:
    """Average historical Market_Adjustment by state - a business-maintained
    lookup table (pricing team's market-positioning assumption per state),
    not something we're predicting."""
    if "Market_Adjustment" in df.columns:
        lookup = df.groupby("State")["Market_Adjustment"].mean().to_dict()
    else:
        lookup = {}
    MARKET_ADJ_PATH.write_text(json.dumps(lookup, indent=2))
    return lookup


def main():
    df = pd.read_parquet(DATA_PATH)
    bundle = get_model_bundle()

    print(f"Scoring {len(df):,} rows...")

    df["Pred_Expected_Claim_Frequency"] = bundle.predict_frequency(df)
    df["Pred_Expected_Severity"] = bundle.predict_severity(df)
    df["Pred_Bind_Probability"] = bundle.predict_bind_probability(df)
    df["Pred_Expected_Loss_Cost"] = df["Pred_Expected_Claim_Frequency"] * df["Pred_Expected_Severity"]

    # Loss_Propensity_Score = percentile rank of Expected_Loss_Cost against the whole book
    df["Loss_Propensity_Score"] = df["Pred_Expected_Loss_Cost"].rank(pct=True) * 100

    market_adj_lookup = build_market_adj_lookup(df)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Applying rules engine"):
        submission = row_to_submission_inputs(row)
        market_adj = market_adj_lookup.get(row["State"], 0.0)
        result = score_submission(
            x=submission,
            expected_claim_frequency=row["Pred_Expected_Claim_Frequency"],
            expected_severity=row["Pred_Expected_Severity"],
            bind_probability=row["Pred_Bind_Probability"],
            loss_propensity_score=row["Loss_Propensity_Score"],
            market_adj=market_adj,
        )
        results.append(result.__dict__)

    results_df = pd.DataFrame(results)
    results_df["hard_stop_reasons"] = results_df["hard_stop_reasons"].apply(lambda r: " | ".join(r) if r else "")

    out = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved model-scored book -> {OUTPUT_PATH}")
    print(f"\nBand distribution:\n{out['uw_risk_band'].value_counts()}")
    print(f"\nAction distribution:\n{out['recommended_action'].value_counts()}")


if __name__ == "__main__":
    main()
