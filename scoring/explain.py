"""
Explainability: the reason codes behind the loss, bind, and appetite scores.

Mirrors the reason-code fields documented (but never populated) in the
source workbook's Data_Dictionary tab: Loss_Reason_Codes, Bind_Reason_Codes,
Appetite_Reason_Codes, Top_5_Reason_Codes / UW_Recommendation,
Compliance_Reason_Codes, Confidence_Level.

Loss and bind reason codes are exact, not approximated:
  - XGBoost and LightGBM both compute true per-instance Shapley-value
    decompositions natively (pred_contribs / pred_contrib) - this is the
    same mechanism the third-party `shap` package wraps for tree models,
    so no extra dependency is needed.
  - The severity model is a linear GLM in link space, so coefficient x
    value is already an exact per-feature contribution, not an estimate.
  - Frequency uses a log link (Poisson) and severity a log link (Gamma),
    so log(frequency) + log(severity) = log(expected loss cost) exactly -
    the two contribution sets are on a common scale and can be summed
    feature-by-feature into one combined Loss Reason Codes list.

Appetite reason codes are an exact decomposition of the deterministic
penalty formula in rules_engine.compute_appetite_score - it's a business
formula, not a model, so there is nothing approximate to explain here
either.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / "rules"))
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from rules_engine import (  # noqa: E402
    SubmissionInputs, ScoringResult, COMPOSITE_THRESHOLDS, NON_NEGOTIABLE_HARD_STOPS,
)
from feature_config import LOSS_FEATURES, LOSS_CATEGORICAL_FEATURES, BIND_FEATURES, BIND_CATEGORICAL_FEATURES  # noqa: E402
from preprocessing import engineer_bind_features  # noqa: E402
from row_mapper import submission_inputs_to_model_row  # noqa: E402

TOP_N = 5


@dataclass
class ReasonCode:
    label: str        # human-readable feature or rule description
    direction: str     # "increases" | "decreases" (impact on risk/propensity)
    weight: float       # relative impact, 0-100, normalized within its list
    contribution: float = 0.0  # signed Shapley contribution in the model's own units


@dataclass
class Waterfall:
    """A Shapley additive decomposition, ready to plot: starting from the
    model's base value, each driver's signed contribution moves the
    prediction to its final value. base + sum(all contributions) == final,
    which is exactly the additivity guarantee Shapley values provide.

    `guardrail_adjustment` is non-zero when the underwriting guardrails in
    predictors.py clipped the model's raw output (frequency to [0.005, 1.5],
    severity to [$500, $100k]). It's surfaced as its own step rather than
    folded away, so the chart still ends on the number the rules engine
    actually used and the clip is visible instead of looking like drift."""
    base_value: float
    final_value: float
    labels: list[str]
    contributions: list[float]
    other_contribution: float  # everything outside the top-N, pooled
    units: str                  # what base/final are measured in
    guardrail_adjustment: float = 0.0


@dataclass
class Explanation:
    loss_reasons: list[ReasonCode]
    bind_reasons: list[ReasonCode]
    appetite_reasons: list[ReasonCode]
    top_reasons: list[ReasonCode]        # combined, priority-ordered digest
    confidence: str                       # "High" | "Medium" | "Low"
    manual_review: bool
    summary: str
    loss_waterfall: Optional[Waterfall] = None
    bind_waterfall: Optional[Waterfall] = None


def _humanize(raw_name: str, categorical_cols: list[str]) -> str:
    """'num__Driver_Age' -> 'Driver Age'; 'cat__Vehicle_Use_Business' ->
    'Vehicle Use: Business' (matched against the known categorical column
    names, since a one-hot name is COLUMN_VALUE with no fixed delimiter).

    A handful of "categorical" columns are actually 0/1 flags that only
    look categorical to the encoder (e.g. Undeclared_Business_Use_Flag),
    so '...Flag: 0' reads as a stray number rather than a fact about the
    submission - phrased as Yes/No instead, and the redundant "Flag" word
    dropped since "...: No" already says the same thing."""
    if raw_name.startswith("num__"):
        return raw_name[len("num__"):].replace("_", " ")
    if raw_name.startswith("cat__"):
        rest = raw_name[len("cat__"):]
        best = max(
            (c for c in categorical_cols if rest == c or rest.startswith(c + "_")),
            key=len, default=None,
        )
        if best is None:
            return rest.replace("_", " ")
        value = rest[len(best):].lstrip("_")
        col_label = best.replace("_", " ")
        if best.endswith("_Flag") and value in ("0", "1"):
            col_label = col_label[:-len(" Flag")]
            value = "Yes" if value == "1" else "No"
        return f"{col_label}: {value}" if value else col_label
    return raw_name.replace("_", " ")


def _raw_contribs(contribs, feature_names: list[str], categorical_cols: list[str],
                   skip_names: frozenset[str] = frozenset()) -> tuple[dict[str, float], float]:
    """Returns (named contributions, total of skipped ones). Skipped
    contributions are returned rather than discarded: they must not appear
    as named reasons (they're computed off imputed placeholders), but they
    still have to be accounted for or the waterfall stops adding up to the
    real prediction - Shapley additivity is the whole guarantee."""
    out: dict[str, float] = {}
    skipped_total = 0.0
    for name, c in zip(feature_names, contribs):
        c = float(c)
        if name in skip_names:
            skipped_total += c
            continue
        if abs(c) < 1e-9:
            continue
        label = _humanize(name, categorical_cols)
        out[label] = out.get(label, 0.0) + c
    return out, skipped_total


def _rank(contrib_map: dict[str, float], top_n: int = TOP_N) -> list[ReasonCode]:
    items = sorted(contrib_map.items(), key=lambda kv: -abs(kv[1]))[:top_n]
    if not items:
        return []
    max_abs = max(abs(v) for _, v in items)
    return [
        ReasonCode(label=label, direction="increases" if v > 0 else "decreases",
                   weight=round(abs(v) / max_abs * 100, 1), contribution=v)
        for label, v in items
    ]


def _build_waterfall(contrib_map: dict[str, float], base_value: float, units: str,
                      link, hidden_total: float = 0.0, top_n: int = TOP_N,
                      actual_final: Optional[float] = None) -> Waterfall:
    """Turn a full contribution map into a plottable additive decomposition.
    Contributions live in the model's LINK space (log space for both the
    Poisson frequency model and the logit bind model), which is where they
    actually sum; `link` maps a link-space total back to the units a person
    reads. Everything below the top-N - plus `hidden_total`, the pooled
    contributions of features excluded from the named reasons - goes into a
    single 'all other factors' bar, so the bars on screen still add up to
    the model's true prediction."""
    items = sorted(contrib_map.items(), key=lambda kv: -abs(kv[1]))
    top = items[:top_n]
    other = sum(v for _, v in items[top_n:]) + hidden_total
    total_link = base_value + sum(contrib_map.values()) + hidden_total
    raw_final = link(total_link)
    final = raw_final if actual_final is None else actual_final
    return Waterfall(
        base_value=link(base_value),
        final_value=final,
        labels=[label for label, _ in top],
        contributions=[v for _, v in top],
        other_contribution=other,
        units=units,
        guardrail_adjustment=final - raw_final,
    )


def explain_loss(bundle, row_df: pd.DataFrame, top_n: int = TOP_N,
                  actual_loss_cost: Optional[float] = None):
    """Combined frequency + severity reason codes, plus a plottable
    waterfall - see module docstring for why summing the two is
    mathematically sound.

    Numeric features that are missing for this submission (e.g. telematics
    behavior metrics for a driver who isn't enrolled) get imputed to the
    training median before scoring - correct for the prediction itself, but
    a contribution computed off an imputed placeholder isn't a genuine
    "reason" about this specific applicant, so those features are excluded
    here rather than presented as if they were observed."""
    unknown_numeric = {
        f"num__{c}" for c in LOSS_FEATURES
        if c in row_df.columns and pd.isna(row_df[c].iloc[0])
    }

    X_freq = bundle.freq_prep.transform(row_df[LOSS_FEATURES])
    freq_all = bundle.freq_model.get_booster().predict(xgb.DMatrix(X_freq), pred_contribs=True)[0]
    freq_contribs, freq_base = freq_all[:-1], float(freq_all[-1])
    freq_map, freq_hidden = _raw_contribs(freq_contribs, list(bundle.freq_prep.get_feature_names_out()),
                                           LOSS_CATEGORICAL_FEATURES, skip_names=unknown_numeric)

    X_sev = bundle.sev_prep.transform(row_df[LOSS_FEATURES])
    x_row = np.asarray(X_sev[0]).ravel()
    coefs = np.asarray(bundle.sev_model.params)  # [const, feat_0, feat_1, ...]
    sev_base = float(coefs[0])
    sev_contribs = coefs[1:len(x_row) + 1] * x_row
    sev_map, sev_hidden = _raw_contribs(sev_contribs, list(bundle.sev_prep.get_feature_names_out()),
                                         LOSS_CATEGORICAL_FEATURES, skip_names=unknown_numeric)

    combined = dict(freq_map)
    for label, v in sev_map.items():
        combined[label] = combined.get(label, 0.0) + v

    # Both models use a log link, so frequency and severity contributions add
    # in log space and exponentiate into an expected loss cost in dollars.
    waterfall = _build_waterfall(
        combined, base_value=freq_base + sev_base, units="$ expected loss cost",
        link=lambda v: float(np.exp(v)), hidden_total=freq_hidden + sev_hidden, top_n=top_n,
        actual_final=actual_loss_cost,
    )
    return _rank(combined, top_n), waterfall


def explain_bind(bundle, row_df: pd.DataFrame, top_n: int = TOP_N,
                  actual_bind_score: Optional[float] = None):
    row_df = engineer_bind_features(row_df)
    X = bundle.bind_prep.transform(row_df[BIND_FEATURES])
    all_contribs = bundle.bind_model.predict(X, pred_contrib=True)[0]
    contribs, base = all_contribs[:-1], float(all_contribs[-1])
    contrib_map, _ = _raw_contribs(contribs, list(bundle.bind_prep.get_feature_names_out()),
                                    BIND_CATEGORICAL_FEATURES)

    # LightGBM binary contributions are in logit space; the sigmoid maps a
    # logit total back to a bind probability.
    waterfall = _build_waterfall(
        contrib_map, base_value=base, units="% bind probability",
        link=lambda v: float(1.0 / (1.0 + np.exp(-v)) * 100.0), top_n=top_n,
        actual_final=actual_bind_score,
    )
    return _rank(contrib_map, top_n), waterfall


def explain_appetite(x: SubmissionInputs, top_n: int = TOP_N) -> list[ReasonCode]:
    """Exact decomposition of compute_appetite_score's penalty terms
    (rules/rules_engine.py) - every penalty point shown here is one of
    that formula's terms, in the same units it uses."""
    terms = [
        ("Prior claims (3Y)", 4 * x.prior_claims_3y),
        ("At-fault claims (3Y)", 5 * x.at_fault_claims_3y),
        ("Moving violations (3Y)", 3 * x.moving_violations_3y),
        ("Major violation on record", 15 if x.major_violation else 0),
        ("Coverage lapse 31-60 days", 10 if 30 < x.coverage_lapse_days <= 60 else 0),
        ("Vehicle value above $80,000", max(0.0, (x.vehicle_value - 80_000) / 2500)),
        ("CAT exposure above 7", max((x.cat_exposure_score - 7) * 4, 0)),
        ("Driver under 25", 5 if x.driver_age < 25 else 0),
        ("Declared business use", 5 if (x.vehicle_use == "Business" and not x.undeclared_business_use) else 0),
        ("Litigation environment above 7", max((x.litigation_environment - 7) * 3, 0)),
        ("3+ prior claims (5Y)", 6 if x.prior_claims_5y >= 3 else 0),
    ]
    penalties = [(label, val) for label, val in terms if val > 0]
    if not penalties:
        return []
    max_val = max(v for _, v in penalties)
    ranked = sorted(penalties, key=lambda t: -t[1])[:top_n]
    return [ReasonCode(label=label, direction="decreases", weight=round(v / max_val * 100, 1))
            for label, v in ranked]


def confidence_level(result: ScoringResult) -> str:
    """How close the composite score sits to a band boundary - a score
    right on a threshold is a lower-confidence call than one deep inside
    a band, independent of which band it landed in."""
    if result.hard_stop:
        return "High"
    gap = min(abs(result.composite_score - t) for t in COMPOSITE_THRESHOLDS.values())
    if gap >= 10:
        return "High"
    if gap >= 4:
        return "Medium"
    return "Low"


def _combine_top_reasons(loss_reasons, bind_reasons, appetite_reasons, top_n: int = TOP_N) -> list[ReasonCode]:
    """Priority-ordered digest: appetite/eligibility first, then loss, then
    bind - mirrors the source workbook's own stated governance ("loss risk
    dominates; bind never drives approval by itself; hard-stops override
    all statistical scores")."""
    combined, seen = [], set()
    for pool in (appetite_reasons, loss_reasons, bind_reasons):
        for r in pool:
            if r.label in seen:
                continue
            seen.add(r.label)
            combined.append(r)
            if len(combined) >= top_n:
                return combined
    return combined


def _plain_english_summary(result: ScoringResult, loss_reasons, bind_reasons, appetite_reasons) -> str:
    if result.hard_stop:
        reasons = "; ".join(result.hard_stop_reasons)
        if all(r in NON_NEGOTIABLE_HARD_STOPS for r in result.hard_stop_reasons):
            return f"Mandatory decline, no discretion available: {reasons}."
        return f"Hard-stop triggered ({reasons}). see Suggestions below for an underwriter alternative."
    clauses = []
    if appetite_reasons:
        clauses.append(f"{appetite_reasons[0].label.lower()} weighing on eligibility")
    if loss_reasons:
        top = loss_reasons[0]
        clauses.append(f"{top.label.lower()} {top.direction} the expected loss cost")
    if bind_reasons:
        top = bind_reasons[0]
        clauses.append(f"{top.label.lower()} {top.direction} the likelihood to bind")
    if not clauses:
        return f"{result.uw_risk_band} band on a clean risk profile. no material adverse drivers found."
    return f"{result.uw_risk_band} band, driven mainly by " + "; ".join(clauses) + "."


def explain_submission(bundle, x: SubmissionInputs, result: ScoringResult) -> Explanation:
    row_df = pd.DataFrame([submission_inputs_to_model_row(x)])
    row_df["Final_Quoted_Premium"] = result.final_quoted_premium

    loss_reasons, loss_waterfall = explain_loss(bundle, row_df, actual_loss_cost=result.expected_loss_cost)
    bind_reasons, bind_waterfall = explain_bind(bundle, row_df, actual_bind_score=result.bind_propensity_score)
    appetite_reasons = explain_appetite(x)
    top_reasons = _combine_top_reasons(loss_reasons, bind_reasons, appetite_reasons)

    return Explanation(
        loss_reasons=loss_reasons,
        bind_reasons=bind_reasons,
        appetite_reasons=appetite_reasons,
        top_reasons=top_reasons,
        confidence=confidence_level(result),
        manual_review=result.recommended_action != "Auto-Quote / STP",
        summary=_plain_english_summary(result, loss_reasons, bind_reasons, appetite_reasons),
        loss_waterfall=loss_waterfall,
        bind_waterfall=bind_waterfall,
    )
