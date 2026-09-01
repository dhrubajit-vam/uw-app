"""
Single source of truth for feature lists.

Enforces the no-leakage rules from the original workbook's governance notes:
  - Loss models (frequency/severity) NEVER see bind/shopping-behavior features.
  - The bind model NEVER sees loss/claims-history features.
  - No model ever sees post-outcome fields (Bound_Flag, Claim_Flag, Claim_Count,
    Incurred_Loss, Earned_Premium, Realized_Loss_Ratio) as an INPUT - those are
    only ever targets.
  - No model sees any other model's *output* columns (Loss_Propensity_Score,
    Bind_Propensity_Score, Composite_UW_Risk_Score, etc.) - those get
    recomputed downstream, not fed back in.
"""

# ---------------------------------------------------------------------------
# Features shared by the two LOSS models (frequency + severity)
# ---------------------------------------------------------------------------
LOSS_NUMERIC_FEATURES = [
    "Driver_Age", "Years_Licensed", "Annual_Mileage",
    "Customer_Tenure_Years",
    "Coverage_Lapse_Days",
    "Prior_Claims_3Y", "At_Fault_Claims_3Y", "Prior_Claims_5Y", "At_Fault_Claims_5Y",
    "Moving_Violations_3Y", "Speeding_Violations_3Y", "Months_Since_Last_Claim",
    "Hard_Braking_Rate", "Rapid_Acceleration_Rate", "Speeding_Rate",
    "Night_Driving_Percentage", "Distracted_Driving_Score", "Telematics_Safety_Score",
    "Vehicle_Age", "Vehicle_Value", "Vehicle_Safety_Rating",
    "Parts_Cost_Index", "Labor_Cost_Index", "Repairability_Score",
    "Traffic_Congestion_Score", "Vehicle_Theft_Risk_Score",
    "Weather_Risk_Score", "Hail_Risk_Score", "Flood_Risk_Score", "Hurricane_Risk_Score",
    "Litigation_Environment_Score", "Repair_Cost_Index", "Medical_Cost_Index",
    "Uninsured_Motorist_Risk", "CAT_Exposure_Score",
    "Collision_Deductible", "Comprehensive_Deductible", "PIP_Limit",
]

LOSS_CATEGORICAL_FEATURES = [
    "New_Renewal_Label", "Multi_Policy_Flag", "State", "Urban_Suburban_Rural",
    "Vehicle_Use", "Marital_Status", "Young_Driver_Flag", "Senior_Driver_Flag",
    "Continuous_Coverage_Flag", "Major_Violation_Flag", "DUI_Flag",
    "License_Suspension_Flag", "Prior_Total_Loss_Flag", "Undeclared_Business_Use_Flag",
    "Telematics_Enrolled", "Body_Type", "EV_Flag", "Luxury_Flag",
    "Performance_Vehicle_Flag", "Anti_Theft_Flag", "ADAS_Flag",
    "No_Fault_State_Flag", "Collision_Flag", "Comprehensive_Flag",
    "PIP_Flag", "UM_UIM_Flag",
]

LOSS_FEATURES = LOSS_NUMERIC_FEATURES + LOSS_CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Features for the BIND model
# ---------------------------------------------------------------------------
BIND_NUMERIC_FEATURES = [
    "Customer_Tenure_Years", "Prior_Year_Premium",
    "Quote_Completion_Minutes", "Quote_Revisions",
    "Agent_Engagement_Score", "Digital_Engagement_Score",
    "Final_Quoted_Premium",          # price actually offered - core driver of bind
    "Premium_Change_Percentage_Input",  # engineered: (Final_Quoted_Premium / Prior_Year_Premium) - 1
]

BIND_CATEGORICAL_FEATURES = [
    "New_Renewal_Label", "Multi_Policy_Flag", "State", "Submission_Channel",
    "Price_Sensitivity_Band",
]

BIND_FEATURES = BIND_NUMERIC_FEATURES + BIND_CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Targets / outcome fields (NEVER used as inputs to any model)
# ---------------------------------------------------------------------------
OUTCOME_FIELDS = [
    "Bound_Flag", "Claim_Flag", "Claim_Count", "Incurred_Loss",
    "Earned_Premium", "Realized_Loss_Ratio",
]

ID_FIELDS = ["Policy_ID", "Customer_ID", "Quote_Date"]
