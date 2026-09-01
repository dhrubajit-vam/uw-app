"""
Underwriter alternative-recommendation engine.

The rules engine's job is to reach a band (Preferred / Standard /
Non-Standard / Decline / Hard Stop). This module's job is different: for
anything that lands in Decline or Hard Stop, a real underwriter rarely just
sends a form-letter decline - they look for a way to write the business.
This module surfaces those alternatives.

Deliberately built the same way as rules_engine.py: deterministic and
re-simulated through the real scoring path wherever the alternative is a
genuine rating lever (deductible, business-use classification, telematics,
non-standard load), so every number shown is auditable, not fabricated.
Only reasons with zero underwriting discretion (DUI, license suspension,
confirmed fraud - see rules_engine.NON_NEGOTIABLE_HARD_STOPS) get no
alternative at all.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "rules"))
from rules_engine import (  # noqa: E402
    COVERAGE_APPETITE_BONUS,
    COVERAGE_OPTIONS,
    NON_NEGOTIABLE_HARD_STOPS,
    ScoringResult,
    SubmissionInputs,
    score_submission,
)
from row_mapper import submission_inputs_to_model_row  # noqa: E402

NON_STANDARD_LOAD = 0.15  # indicative price-up load for non-standard placement
DEDUCTIBLE_STEP = 500
MIN_DEDUCTIBLE = 1000


def score_live_submission(bundle, x: SubmissionInputs, book: pd.DataFrame,
                           market_adj_lookup: dict, non_standard_load: float = 0.0) -> ScoringResult:
    """Score a single hypothetical submission end-to-end: models -> rules
    engine. Used for the primary Interface-tab submit AND to re-simulate
    every alternative below against the same deterministic path.

    Two-pass because the bind model needs Final_Quoted_Premium, which isn't
    known until after a first pass through the rules engine: pass 1 with a
    placeholder bind probability computes the premium, pass 2 re-scores
    with the real bind probability."""
    row_df = pd.DataFrame([submission_inputs_to_model_row(x)])

    # Cast off numpy scalar types (XGBoost in particular returns float32,
    # which isn't JSON-serializable - it would otherwise leak into every
    # downstream ScoringResult field and break st.json's calculation-detail
    # panel) so everything downstream is a plain Python float.
    freq_pred = float(bundle.predict_frequency(row_df).iloc[0])
    sev_pred = float(bundle.predict_severity(row_df).iloc[0])
    expected_loss_cost = freq_pred * sev_pred
    loss_propensity_score = float((book["Pred_Expected_Loss_Cost"] < expected_loss_cost).mean() * 100)
    market_adj = market_adj_lookup.get(x.state, 0.0)

    prelim = score_submission(
        x=x, expected_claim_frequency=freq_pred, expected_severity=sev_pred,
        bind_probability=0.5, loss_propensity_score=loss_propensity_score,
        market_adj=market_adj, non_standard_load=non_standard_load,
    )
    row_df["Final_Quoted_Premium"] = prelim.final_quoted_premium
    bind_prob = float(bundle.predict_bind_probability(row_df).iloc[0])

    return score_submission(
        x=x, expected_claim_frequency=freq_pred, expected_severity=sev_pred,
        bind_probability=bind_prob, loss_propensity_score=loss_propensity_score,
        market_adj=market_adj, non_standard_load=non_standard_load,
    )


_BAND_RANK = {"Preferred": 3, "Standard": 2, "Non-Standard": 1, "Decline": 0, "Hard Stop": 0}

# A coverage switch is only worth putting in front of an underwriter if it
# changes the band, or moves the composite by more than scoring noise. A
# fraction of a point is not a recommendation.
MIN_COMPOSITE_GAIN = 2.0
MIN_PREMIUM_GAIN = 25.0


@dataclass
class CoverageFit:
    """One coverage option scored end-to-end against the same risk."""
    coverage_type: str
    appetite_bonus: float
    appetite_score: float
    composite_score: float
    uw_risk_band: str
    technical_premium: float
    final_quoted_premium: float
    is_current: bool


@dataclass
class CoverageAdvice:
    options: list[CoverageFit]
    current: CoverageFit
    best: CoverageFit
    improves: bool     # does the best option beat the current band?
    message: str


def coverage_fit_analysis(bundle, x: SubmissionInputs, book: pd.DataFrame,
                           market_adj_lookup: dict) -> CoverageAdvice:
    """Score the SAME risk under every coverage option.

    Coverage isn't just an appetite adjustment here: it flips the actual
    Collision / Comprehensive / UM-UIM flags the loss models read, so each
    option gets its own predicted loss cost, technical premium, and quoted
    premium alongside its appetite and composite score. That's what makes
    this a coverage RECOMMENDATION ("write it this way instead") rather
    than just a scorecard."""
    fits: list[CoverageFit] = []
    for option in COVERAGE_OPTIONS:
        xi = replace(x, coverage_type=option)
        r = score_live_submission(bundle, xi, book, market_adj_lookup)
        fits.append(CoverageFit(
            coverage_type=option,
            appetite_bonus=COVERAGE_APPETITE_BONUS.get(option, 0),
            appetite_score=r.appetite_score,
            composite_score=r.composite_score,
            uw_risk_band=r.uw_risk_band,
            technical_premium=r.technical_premium,
            final_quoted_premium=r.final_quoted_premium,
            is_current=(option == x.coverage_type),
        ))

    current = next(f for f in fits if f.is_current)
    # Rank on band first (a better band is a materially different
    # underwriting outcome, not just a higher number), then on technical
    # premium: among options tied on band, prefer the one that carries the
    # highest technical premium - maximizing premium written for the same
    # risk tier, per underwriting policy.
    best = max(fits, key=lambda f: (_BAND_RANK.get(f.uw_risk_band, 0), f.technical_premium))

    band_gain = _BAND_RANK.get(best.uw_risk_band, 0) - _BAND_RANK.get(current.uw_risk_band, 0)
    composite_gain = best.composite_score - current.composite_score
    premium_delta = best.final_quoted_premium - current.final_quoted_premium

    # A hard-stop rule fires on eligibility, not on coverage form, so no
    # coverage switch can clear it. Never dangle one as if it could.
    hard_stopped = current.uw_risk_band == "Hard Stop"

    improves = (not hard_stopped) and best.coverage_type != current.coverage_type and (
        band_gain > 0 or composite_gain >= MIN_COMPOSITE_GAIN
    )

    if hard_stopped:
        message = ("A hard-stop rule is in force on this submission. Coverage selection cannot "
                   "clear it, the rule applies whichever coverage form is written. See the "
                   "underwriter recommendations below for the routes that do apply.")
    elif improves:
        if abs(premium_delta) < MIN_PREMIUM_GAIN:
            # Quoted premium can be pinned by a guardrail (renewal band,
            # floor/ceiling) even when the technical premium moves, so quote
            # the technical difference instead of claiming a saving of zero.
            tech_delta = best.technical_premium - current.technical_premium
            price_clause = (f"quoted premium effectively unchanged (a pricing guardrail is binding), "
                            f"though technical premium moves ${abs(tech_delta):,.0f} "
                            f"{'up' if tech_delta > 0 else 'down'}")
        else:
            price_clause = (f"quoted premium ${best.final_quoted_premium:,.0f}, "
                            f"${abs(premium_delta):,.0f} "
                            f"{'higher' if premium_delta > 0 else 'lower'}")
        band_clause = (f"band {current.uw_risk_band} becomes {best.uw_risk_band}"
                       if band_gain > 0 else f"stays in {best.uw_risk_band}")
        message = (f"Consider writing this as {best.coverage_type}. Same risk: composite "
                   f"{current.composite_score:.1f} becomes {best.composite_score:.1f}, "
                   f"{band_clause}, {price_clause}.")
    elif _BAND_RANK.get(current.uw_risk_band, 0) == 0:
        message = ("No coverage option resolves this outcome. It's driven by loss-cost and "
                   "eligibility factors that apply regardless of which coverage form is written.")
    else:
        message = (f"{current.coverage_type} is the right fit for this risk. No other coverage "
                   f"option produces a better band or a materially better score.")

    return CoverageAdvice(options=fits, current=current, best=best,
                           improves=improves, message=message)


@dataclass
class Alternative:
    category: str            # "Modify Terms" | "Price-Up / Non-Standard Placement" | "Conditional Acceptance"
    title: str
    change_summary: str
    result: Optional[ScoringResult] = None  # None for referral-only alternatives that can't be re-rated here
    note: Optional[str] = None              # extra process note (sign-off required, etc.)


def _hard_stop_alternatives(bundle, x: SubmissionInputs, book, market_adj_lookup,
                             negotiable_reasons: list[str]) -> list[Alternative]:
    alts: list[Alternative] = []

    if "Undeclared business use" in negotiable_reasons:
        x2 = replace(x, undeclared_business_use=False, vehicle_use="Business")
        alts.append(Alternative(
            category="Modify Terms",
            title="Reclassify to declared business use",
            change_summary="Vehicle use corrected from undeclared to declared Business use and re-rated "
                            "accordingly, removing the misrepresentation hard-stop.",
            result=score_live_submission(bundle, x2, book, market_adj_lookup),
        ))

    if "Coverage lapse exceeds 60 days" in negotiable_reasons:
        alts.append(Alternative(
            category="Conditional Acceptance",
            title="Accept subject to a monitored lapse period",
            change_summary="Signed lapse affidavit plus a 90-day claims-free monitoring window; "
                            "standard terms apply at the next review if the period is clean.",
            note="Requires senior underwriter sign-off before bind.",
        ))

    if "Vehicle value exceeds carrier limit ($100,000)" in negotiable_reasons:
        alts.append(Alternative(
            category="Price-Up / Non-Standard Placement",
            title="Refer to high-value / specialty auto placement",
            change_summary=f"Vehicle value (${x.vehicle_value:,.0f}) exceeds this carrier's $100,000 "
                            f"standard-market limit. Route to an agreed-value specialty partner "
                            f"instead of a flat decline.",
        ))

    if "State outside carrier appetite footprint" in negotiable_reasons:
        alts.append(Alternative(
            category="Price-Up / Non-Standard Placement",
            title="Refer to excess & surplus (E&S) placement",
            change_summary=f"{x.state} is outside this carrier's standard appetite footprint. "
                            f"Route to a non-admitted / E&S partner rather than declining outright.",
        ))

    if "Prior total-loss claim" in negotiable_reasons:
        alts.append(Alternative(
            category="Conditional Acceptance",
            title="Manual override after vehicle inspection",
            change_summary="Refer to senior underwriter for override consideration after a physical "
                            "inspection/appraisal, bound with a comprehensive-only endorsement and an "
                            "elevated deductible.",
            note="Requires senior underwriter sign-off before bind.",
        ))

    return alts


def _decline_alternatives(bundle, x: SubmissionInputs, base_result: ScoringResult, book,
                           market_adj_lookup: dict) -> list[Alternative]:
    alts: list[Alternative] = []

    claims_or_young_driver = (
        x.prior_claims_3y > 0 or x.at_fault_claims_3y > 0
        or x.moving_violations_3y > 0 or x.driver_age < 25
    )
    if claims_or_young_driver and not x.telematics_enrolled:
        x_telem = replace(x, telematics_enrolled=True,
                           telematics_safety_score=max(x.telematics_safety_score, 70))
        alts.append(Alternative(
            category="Conditional Acceptance",
            title="Require telematics enrollment",
            change_summary="Accept conditional on telematics enrollment with an ongoing safe-driving "
                            "score requirement, re-evaluated at the next renewal.",
            result=score_live_submission(bundle, x_telem, book, market_adj_lookup),
        ))

    high_value_or_cat = x.vehicle_value >= 40_000 or x.cat_exposure_score >= 6
    low_deductibles = x.collision_deductible < MIN_DEDUCTIBLE or x.comprehensive_deductible < MIN_DEDUCTIBLE
    if high_value_or_cat and low_deductibles:
        new_collision = max(MIN_DEDUCTIBLE, x.collision_deductible + DEDUCTIBLE_STEP)
        new_comp = max(MIN_DEDUCTIBLE, x.comprehensive_deductible + DEDUCTIBLE_STEP)
        x_ded = replace(x, collision_deductible=new_collision, comprehensive_deductible=new_comp)
        alts.append(Alternative(
            category="Modify Terms",
            title="Raise collision / comprehensive deductibles",
            change_summary=f"Deductibles increased to ${new_collision:,.0f} collision / "
                            f"${new_comp:,.0f} comprehensive to bring the technical premium and "
                            f"risk load into an acceptable range.",
            result=score_live_submission(bundle, x_ded, book, market_adj_lookup),
        ))

    # Always available: price up instead of declining outright.
    alts.append(Alternative(
        category="Price-Up / Non-Standard Placement",
        title=f"Quote at a {NON_STANDARD_LOAD:.0%} non-standard load",
        change_summary="Same terms, priced for the elevated risk through the non-standard tier "
                        "rather than declining the submission.",
        result=score_live_submission(bundle, x, book, market_adj_lookup, non_standard_load=NON_STANDARD_LOAD),
    ))

    return alts


def build_alternatives(bundle, x: SubmissionInputs, base_result: ScoringResult, book: pd.DataFrame,
                        market_adj_lookup: dict) -> list[Alternative]:
    """Return up to 3 underwriter alternatives to a Decline/Hard-Stop
    outcome, ranked with the most impactful re-simulated option first.
    Empty list means there is genuinely no alternative to a mandatory
    decline (every triggered hard-stop reason is non-negotiable)."""
    if base_result.hard_stop:
        negotiable = [r for r in base_result.hard_stop_reasons if r not in NON_NEGOTIABLE_HARD_STOPS]
        if not negotiable:
            return []
        alts = _hard_stop_alternatives(bundle, x, book, market_adj_lookup, negotiable)
    elif base_result.uw_risk_band == "Decline":
        alts = _decline_alternatives(bundle, x, base_result, book, market_adj_lookup)
    else:
        return []

    def sort_key(a: Alternative):
        if a.result is not None:
            return (0, -(a.result.composite_score - base_result.composite_score))
        return (1, 0)

    alts.sort(key=sort_key)
    return alts[:3]
