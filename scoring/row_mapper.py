"""
Maps a raw data row (dict-like: dataframe row or Streamlit form output)
into the rules_engine.SubmissionInputs dataclass. Single source of truth
for the column-name -> field-name mapping, used by both score_book.py
(historical data) and app.py (live form input).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rules"))
from rules_engine import SubmissionInputs, COVERAGE_FLAGS, DEFAULT_COVERAGE  # noqa: E402


# Proxy/default values for model features the live Interface form doesn't
# collect (either genuinely unavailable pre-bind, or intentionally kept out
# of the form to keep it usable). Centralized here - not buried inline in
# the UI - so the live-scoring schema can't silently drift from
# score_book.py's.

# Fields the Interface form doesn't collect. Every one of these is genuinely
# UNKNOWN for a live submission, so it's passed as missing rather than as a
# hand-picked stand-in value: each model's preprocessor already carries a
# SimpleImputer fitted on the training data (median for numerics,
# most-frequent for categoricals), so a missing value resolves to exactly the
# population-typical value the models were trained against, and stays correct
# automatically whenever the models are retrained.
#
# This matters more than it looks. Hand-picked "middle of the scale" defaults
# were measurably wrong here: Weather/Hail/Flood/Hurricane risk all default to
# a 0-10 midpoint of 5, but the book's medians are ~1.9-3.2, and
# Months_Since_Last_Claim was set to 999 against a real median of 61. Those
# pushed every live submission's predicted loss cost far above the book, so a
# genuinely clean driver scored at the 99th percentile of loss risk.
_UNKNOWN = float("nan")

_MODEL_ROW_DEFAULTS = {
    "Months_Since_Last_Claim": _UNKNOWN,
    "Hard_Braking_Rate": _UNKNOWN,
    "Rapid_Acceleration_Rate": _UNKNOWN,
    "Speeding_Rate": _UNKNOWN,
    "Night_Driving_Percentage": _UNKNOWN,
    "Distracted_Driving_Score": _UNKNOWN,
    "Vehicle_Age": _UNKNOWN, "Vehicle_Safety_Rating": _UNKNOWN,
    "Weather_Risk_Score": _UNKNOWN, "Hail_Risk_Score": _UNKNOWN,
    "Flood_Risk_Score": _UNKNOWN, "Hurricane_Risk_Score": _UNKNOWN,
    "Repair_Cost_Index": _UNKNOWN, "Uninsured_Motorist_Risk": _UNKNOWN,
    "PIP_Limit": _UNKNOWN,
    "Marital_Status": None,
    "Body_Type": None, "EV_Flag": _UNKNOWN,
    "ADAS_Flag": _UNKNOWN, "No_Fault_State_Flag": _UNKNOWN,
    "PIP_Flag": _UNKNOWN,
    "Quote_Completion_Minutes": _UNKNOWN,
}


def _cov(x: SubmissionInputs) -> dict:
    return COVERAGE_FLAGS.get(x.coverage_type, COVERAGE_FLAGS[DEFAULT_COVERAGE])


def submission_inputs_to_model_row(x: SubmissionInputs) -> dict:
    """Forward mapping: SubmissionInputs (the Interface form / a what-if
    variant of it) -> the flat feature row the trained models expect. The
    mirror of row_to_submission_inputs() below. Single source of truth so
    app.py's live form and scoring/advisor.py's what-if resimulations never
    drift from score_book.py's schema."""
    row = dict(_MODEL_ROW_DEFAULTS)
    row.update({
        "Driver_Age": x.driver_age, "Years_Licensed": x.years_licensed,
        "Annual_Mileage": x.annual_mileage,
        "Customer_Tenure_Years": x.customer_tenure_years,
        "Coverage_Lapse_Days": x.coverage_lapse_days,
        "Prior_Claims_3Y": x.prior_claims_3y, "At_Fault_Claims_3Y": x.at_fault_claims_3y,
        "Prior_Claims_5Y": x.prior_claims_5y,
        # 5-year at-fault detail isn't collected by the form; the 3-year
        # figure is used as a conservative proxy.
        "At_Fault_Claims_5Y": x.at_fault_claims_3y,
        "Moving_Violations_3Y": x.moving_violations_3y,
        "Speeding_Violations_3Y": x.speeding_violations_3y,
        # Null (imputed to the training median) unless actually enrolled -
        # a score can't exist for a driver who was never measured, so the
        # slider value is only meaningful when Telematics Enrolled is checked.
        "Telematics_Safety_Score": x.telematics_safety_score if x.telematics_enrolled else float("nan"),
        "Vehicle_Value": x.vehicle_value,
        "Parts_Cost_Index": x.parts_cost_index, "Labor_Cost_Index": x.labor_cost_index,
        "Repairability_Score": x.repairability_score,
        "Traffic_Congestion_Score": x.traffic_congestion,
        "Vehicle_Theft_Risk_Score": x.vehicle_theft_risk,
        "Body_Type": x.body_type,
        "Anti_Theft_Flag": 1 if x.anti_theft_available else 0,
        "Litigation_Environment_Score": x.litigation_environment,
        "Medical_Cost_Index": x.medical_cost_index,
        "CAT_Exposure_Score": x.cat_exposure_score,
        "Collision_Deductible": x.collision_deductible,
        "Comprehensive_Deductible": x.comprehensive_deductible,
        # Coverage selection drives the actual coverage flags the loss models
        # read, so switching coverage moves the predicted loss cost (and
        # therefore the technical premium), not just the appetite score.
        "Collision_Flag": _cov(x)["collision"],
        "Comprehensive_Flag": _cov(x)["comprehensive"],
        "UM_UIM_Flag": _cov(x)["um_uim"],
        "New_Renewal_Label": x.new_renewal,
        "Multi_Policy_Flag": 1 if x.multi_policy else 0,
        "State": x.state, "Urban_Suburban_Rural": x.territory_type, "Vehicle_Use": x.vehicle_use,
        "Young_Driver_Flag": 1 if x.driver_age < 25 else 0,
        "Senior_Driver_Flag": 1 if x.driver_age >= 65 else 0,
        "Continuous_Coverage_Flag": 1 if x.coverage_lapse_days == 0 else 0,
        "Major_Violation_Flag": 1 if x.major_violation else 0,
        "DUI_Flag": 1 if x.dui_flag else 0,
        "License_Suspension_Flag": 1 if x.license_suspension else 0,
        "Prior_Total_Loss_Flag": 1 if x.prior_total_loss else 0,
        "Undeclared_Business_Use_Flag": 1 if x.undeclared_business_use else 0,
        "Telematics_Enrolled": 1 if x.telematics_enrolled else 0,
        "Luxury_Flag": 1 if x.luxury_vehicle else 0,
        "Performance_Vehicle_Flag": 1 if x.performance_vehicle else 0,
        # bind-model-only fields
        "Prior_Year_Premium": x.prior_year_premium if x.prior_year_premium else 1200,
        "Quote_Revisions": x.quote_revisions,
        # Not collected by the Interface form - genuinely unknown pre-bind,
        # so left missing rather than guessed (see SubmissionInputs).
        "Agent_Engagement_Score": x.agent_engagement if x.agent_engagement is not None else _UNKNOWN,
        "Digital_Engagement_Score": x.digital_engagement if x.digital_engagement is not None else _UNKNOWN,
        "Price_Sensitivity_Band": x.price_sensitivity,
        "Submission_Channel": x.submission_channel,
    })
    return row


def _yn(v) -> bool:
    """Handles flags stored as bool, numeric 0/1, or 'Y'/'N' strings."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().upper() in ("Y", "1", "TRUE")


def row_to_submission_inputs(row, target_profit_margin: float = 0.05,
                              competitor_premium_estimate=None) -> SubmissionInputs:
    return SubmissionInputs(
        driver_age=float(row["Driver_Age"]),
        years_licensed=float(row["Years_Licensed"]),
        annual_mileage=float(row["Annual_Mileage"]),
        vehicle_use=row["Vehicle_Use"],
        territory_type=row["Urban_Suburban_Rural"],
        traffic_congestion=float(row["Traffic_Congestion_Score"]),
        vehicle_theft_risk=float(row["Vehicle_Theft_Risk_Score"]),
        coverage_lapse_days=float(row["Coverage_Lapse_Days"]),
        undeclared_business_use=_yn(row["Undeclared_Business_Use_Flag"]),
        prior_claims_3y=int(row["Prior_Claims_3Y"]),
        at_fault_claims_3y=int(row["At_Fault_Claims_3Y"]),
        prior_claims_5y=int(row["Prior_Claims_5Y"]),
        moving_violations_3y=int(row["Moving_Violations_3Y"]),
        speeding_violations_3y=int(row["Speeding_Violations_3Y"]),
        major_violation=_yn(row["Major_Violation_Flag"]),
        dui_flag=_yn(row["DUI_Flag"]),
        license_suspension=_yn(row["License_Suspension_Flag"]),
        telematics_enrolled=_yn(row["Telematics_Enrolled"]),
        telematics_safety_score=float(row["Telematics_Safety_Score"]) if row["Telematics_Safety_Score"] is not None else 75.0,
        vehicle_value=float(row["Vehicle_Value"]),
        repairability_score=float(row["Repairability_Score"]),
        parts_cost_index=float(row["Parts_Cost_Index"]),
        labor_cost_index=float(row["Labor_Cost_Index"]),
        medical_cost_index=float(row["Medical_Cost_Index"]),
        litigation_environment=float(row["Litigation_Environment_Score"]),
        luxury_vehicle=_yn(row["Luxury_Flag"]),
        performance_vehicle=_yn(row["Performance_Vehicle_Flag"]),
        collision_deductible=float(row["Collision_Deductible"]) if row["Collision_Deductible"] is not None else 500.0,
        comprehensive_deductible=float(row["Comprehensive_Deductible"]) if row["Comprehensive_Deductible"] is not None else 500.0,
        cat_exposure_score=float(row["CAT_Exposure_Score"]),
        state_in_appetite=_yn(row["Carrier_Appetite_State_Flag"]),
        prior_total_loss=_yn(row["Prior_Total_Loss_Flag"]),
        confirmed_fraud=_yn(row["Fraud_Flag"]),
        new_renewal=row["New_Renewal_Label"],
        customer_tenure_years=float(row["Customer_Tenure_Years"]),
        multi_policy=_yn(row["Multi_Policy_Flag"]),
        submission_channel=row["Submission_Channel"],
        state=row["State"],
        quote_revisions=int(row["Quote_Revisions"]) if row["Quote_Revisions"] is not None else 0,
        competing_quotes_count=0,
        price_sensitivity=row["Price_Sensitivity_Band"] if row.get("Price_Sensitivity_Band") is not None else "Medium",
        competitor_premium_estimate=competitor_premium_estimate,
        prior_year_premium=float(row["Prior_Year_Premium"]) if row["Prior_Year_Premium"] not in (None,) else None,
        target_profit_margin=target_profit_margin,
        agent_engagement=float(row["Agent_Engagement_Score"]) if row.get("Agent_Engagement_Score") is not None else None,
        digital_engagement=float(row["Digital_Engagement_Score"]) if row.get("Digital_Engagement_Score") is not None else None,
        anti_theft_available=_yn(row["Anti_Theft_Flag"]) if row.get("Anti_Theft_Flag") is not None else False,
        body_type=row.get("Body_Type"),
    )
