"""
US Personal Auto Underwriting & Pricing - Streamlit app (model-backed).

Two tabs:
  - Dashboard: book-wide KPIs, filters, charts.
  - Interface: live risk scoring, with suggestions (underwriter
    alternatives to a flat decline) and explainability (reason codes
    behind the loss, bind, and appetite scores) once you click
    "Calculate Risk Score".

Run:
    streamlit run app/app.py
"""
import base64
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
sys.path.insert(0, str(ROOT / "rules"))
sys.path.insert(0, str(ROOT / "train"))

from predictors import get_model_bundle  # noqa: E402
from rules_engine import (  # noqa: E402
    SubmissionInputs, COVERAGE_OPTIONS, DEFAULT_COVERAGE, RENEWAL_BAND_LOW, RENEWAL_BAND_HIGH,
)
from advisor import score_live_submission, build_alternatives, coverage_fit_analysis  # noqa: E402
from explain import explain_submission  # noqa: E402
from scenarios import SCENARIOS, SCENARIO_NAMES, BASE as SCENARIO_BASE  # noqa: E402
import ai_narrator  # noqa: E402

APP_TITLE = "US Personal Auto Underwriting & Pricing"
APP_SUBTITLE = "Model-backed risk scoring, pricing, and underwriter recommendations."

# Oldest Driver_Age in the training book (data/scored_dataset_full.parquet).
# The input above this still accepts a real age, since a licensed driver
# this old genuinely exists, but the models have never seen one: past this
# point they're extrapolating a learned pattern, not applying a calibrated
# one, so the UI flags it rather than presenting the result as equally
# trustworthy.
MAX_TRAINED_DRIVER_AGE = 85

DATA_PATH = ROOT / "data" / "model_scored_dataset.parquet"
MARKET_ADJ_PATH = ROOT / "models" / "market_adj_by_state.json"
THEFT_RISK_PATH = ROOT / "models" / "theft_risk_by_state.json"
ANTI_THEFT_REDUCTION = 1.5  # points off the state baseline when an anti-theft device is present
BODY_TYPE_OPTIONS = ["Sedan", "SUV", "Truck", "Minivan", "Sports"]
LOGO_PATH = ROOT / "assets" / "logo.png"  # drop your company logo here (PNG/SVG, transparent bg)

# State-based auto-populated cost factors for the Driver & Loss Risk section
# (Labor Cost Index, Medical Cost Index, Litigation/Liability Environment).
# NOTE: Litigation Environment here is on the same ~0.85-1.30 multiplier
# scale as the cost indices, NOT the original 0-10 scale that
# rules_engine.compute_appetite_score's litigation penalty term and the
# trained loss models' Litigation_Environment_Score feature expect - this
# field's effective scale changed when it became state-driven instead of a
# manually-entered 0-10 slider. The appetite penalty term (which only fires
# above 7) is effectively dormant for every auto-populated value below.
STATE_FACTORS = {
    "CA": {"labor_cost_index": 1.20, "medical_cost_index": 1.15, "litigation_environment": 1.20},
    "NY": {"labor_cost_index": 1.25, "medical_cost_index": 1.20, "litigation_environment": 1.25},
    "FL": {"labor_cost_index": 1.05, "medical_cost_index": 1.10, "litigation_environment": 1.30},
    "GA": {"labor_cost_index": 1.00, "medical_cost_index": 1.00, "litigation_environment": 1.15},
    "IL": {"labor_cost_index": 1.05, "medical_cost_index": 1.05, "litigation_environment": 1.10},
    "PA": {"labor_cost_index": 1.05, "medical_cost_index": 1.05, "litigation_environment": 1.05},
    "TX": {"labor_cost_index": 0.95, "medical_cost_index": 0.95, "litigation_environment": 1.05},
    "AZ": {"labor_cost_index": 0.95, "medical_cost_index": 0.95, "litigation_environment": 0.95},
    "OH": {"labor_cost_index": 0.90, "medical_cost_index": 0.90, "litigation_environment": 0.90},
    "NC": {"labor_cost_index": 0.92, "medical_cost_index": 0.92, "litigation_environment": 0.85},
}
# Neutral fallback for the many book states outside the 10-state table above,
# kept on the same ~1.0-centered multiplier scale rather than mixing in the
# old 0-10 default.
DEFAULT_STATE_FACTORS = {"labor_cost_index": 1.0, "medical_cost_index": 1.0, "litigation_environment": 1.0}


def get_state_factors(state: str) -> dict:
    return STATE_FACTORS.get(state, DEFAULT_STATE_FACTORS)


# Reset restores every Interface input to its blank/zero state (not the demo
# scenario's values) - selects use their first/most-neutral option since a
# selectbox has no true "zero", and fields with a hard non-zero minimum
# (driver_age, vehicle_value, parts_cost_index) use that minimum.
RESET_DEFAULTS = {
    "existing_customer": False,
    "coverage_type": DEFAULT_COVERAGE,
    # 1. Driver & Loss Risk Parameters
    "driver_age": 16, "years_licensed": 0, "annual_mileage": 0,
    "state": "TX", "territory_type": "Suburban", "vehicle_use": "Pleasure",
    "body_type": "Sedan", "coverage_lapse_days": 0, "vehicle_value": 500,
    "anti_theft_available": False, "collision_deductible": 0, "comprehensive_deductible": 0,
    "repairability_score": 0, "parts_cost_index": 0.5,
    "undeclared_business_use": False, "luxury_vehicle": False,
    # 2. Claims History
    "prior_claims_3y": 0, "at_fault_claims_3y": 0, "prior_claims_5y": 0,
    "moving_violations_3y": 0, "speeding_violations_3y": 0,
    "major_violation": False, "dui_flag": False, "license_suspension": False,
    # 3. Telematics
    "telematics_enrolled": False, "hard_braking_events": 0,
    "night_driving_pct": 0, "speeding_events_month": 0,
    # 4. Appetite & Eligibility
    "cat_exposure_score": 0, "state_in_appetite": False,
    "prior_total_loss": False, "confirmed_fraud": False,
    # 5. Bind & Market
    "customer_tenure_years": 0, "multi_policy": False, "submission_channel": "Direct",
    "quote_revisions": 0, "price_sensitivity": "Low",
    "competitor_premium_estimate": 0, "prior_year_premium": 0,
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# THEME / COLOR PALETTE
# ============================================================================
NAVY = "#0B2545"
NAVY_LIGHT = "#13315C"
ACCENT = "#2E86FF"
ACCENT_SOFT = "#E8F1FF"
TEAL = "#0EA5A4"
AMBER = "#F5A623"
RED = "#E5484D"
GREEN = "#2FAE6B"
# Standard SHAP diverging pair, used only for the SHAP-driven driver charts
# (Loss Risk / Bind Propensity Drivers) - matches the red/blue convention of
# the `shap` library's own plots (red = pushes the prediction up, blue =
# pushes it down), rather than this app's other red/green good-bad palette.
SHAP_RED = "#FF0D57"
SHAP_BLUE = "#1E88E5"
GRAY_BG = "#F5F7FA"
TEXT_MUTED = "#5A6B87"

BAND_COLORS = {
    "Preferred": GREEN,
    "Standard": ACCENT,
    "Non-Standard": AMBER,
    "Decline": RED,
    "Hard Stop": "#8A1C1C",
}

# The rules engine's actual action strings are long (they're written to be
# unambiguous in an audit trail), so the dashboard displays a shorter label
# for each one. Full name is still one hover away.
ACTION_SHORT_LABELS = {
    "Auto-Quote / STP": "Auto-Quote / STP",
    "Light UW Review": "Light UW Review",
    "Thorough Verification": "Thorough Verification",
    "Refer to Underwriter (Rate-Adequacy Exception)": "Refer (Rate Adequacy)",
    "Refer to Senior Underwriter": "Refer to Senior UW",
    "Decline": "Decline",
    "Hard-Stop Route (Decline / Senior UW Review per triggered rule)": "Hard-Stop Route",
}

ACTION_COLORS = {
    "Auto-Quote / STP": GREEN,
    "Light UW Review": ACCENT,
    "Thorough Verification": AMBER,
    "Refer (Rate Adequacy)": TEAL,
    "Refer to Senior UW": NAVY_LIGHT,
    "Decline": RED,
    "Hard-Stop Route": "#8A1C1C",
}

RECOMMENDATION_COLORS = {
    "Modify Terms": ACCENT,
    "Price-Up / Non-Standard Placement": AMBER,
    "Conditional Acceptance": TEAL,
}

AI_ACCENT = "#7C3AED"

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, 'Segoe UI', sans-serif", color=NAVY, size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[ACCENT, TEAL, AMBER, RED, GREEN, NAVY_LIGHT],
        title=dict(font=dict(size=16, color=NAVY)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#EEF1F6", zeroline=False, linecolor="#D8DEE9"),
        yaxis=dict(gridcolor="#EEF1F6", zeroline=False, linecolor="#D8DEE9"),
        margin=dict(t=50, l=10, r=10, b=10),
    )
)
px.defaults.template = PLOTLY_TEMPLATE

# ============================================================================
# GLOBAL CSS
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    .stApp {{
        background-color: {GRAY_BG};
    }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}

    /* ---- Top brand bar ---- */
    .brand-bar {{
        display: flex;
        align-items: center;
        gap: 20px;
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
        padding: 16px 28px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 18px rgba(11,37,69,0.18);
    }}
    .brand-logo {{
        width: 72px;
        height: 72px;
        border-radius: 14px;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        font-weight: 800;
        color: {NAVY};
        flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }}
    .brand-logo img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 14px;
        padding: 6px;
    }}
    .brand-text h1 {{
        color: white;
        font-size: 25px;
        font-weight: 800;
        margin: 0;
        line-height: 1.25;
    }}
    .brand-text p {{
        color: #C6D4EA;
        font-size: 13px;
        margin: 4px 0 0 0;
    }}

    /* ---- KPI cards ---- */
    .kpi-card {{
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(11,37,69,0.08);
        border: 1px solid #EAEEF4;
        height: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(11,37,69,0.12);
    }}
    .kpi-label {{
        font-size: 12px;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 24px;
        font-weight: 800;
        color: {NAVY};
    }}
    .kpi-sub {{
        font-size: 12px;
        color: {TEXT_MUTED};
        margin-top: 2px;
    }}

    /* ---- Section headers ---- */
    .section-title {{
        font-size: 15px;
        font-weight: 700;
        color: {NAVY};
        margin: 6px 0 10px 0;
        padding-left: 10px;
        border-left: 4px solid {ACCENT};
    }}
    .section-sub {{
        color: {TEXT_MUTED};
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 14px;
    }}

    /* ---- Chart card wrapper ---- */
    .chart-card {{
        background: white;
        border-radius: 12px;
        padding: 14px 16px 4px 16px;
        border: 1px solid #EAEEF4;
        box-shadow: 0 1px 3px rgba(11,37,69,0.06);
        margin-bottom: 18px;
        transition: box-shadow 0.15s ease;
    }}
    .chart-card:hover {{
        box-shadow: 0 6px 18px rgba(11,37,69,0.1);
    }}

    /* ---- Result banner ---- */
    .result-banner {{
        border-radius: 14px;
        padding: 20px 24px;
        color: white;
        margin: 8px 0 18px 0;
    }}
    .band-dot {{
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: white;
        margin-right: 8px;
        vertical-align: middle;
    }}

    /* ---- Recommendation cards ---- */
    .rec-card {{
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid #EAEEF4;
        box-shadow: 0 1px 3px rgba(11,37,69,0.06);
        margin-bottom: 14px;
    }}
    .rec-badge {{
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
    }}
    .rec-title {{
        font-size: 16px;
        font-weight: 700;
        color: {NAVY};
        margin-bottom: 6px;
    }}
    .rec-summary {{
        font-size: 13px;
        color: {TEXT_MUTED};
        line-height: 1.5;
        margin-bottom: 10px;
    }}
    .rec-impact {{
        display: flex;
        gap: 22px;
        flex-wrap: wrap;
        font-size: 13px;
        color: {NAVY};
        border-top: 1px dashed #EAEEF4;
        padding-top: 10px;
    }}
    .rec-note {{
        font-size: 12px;
        color: {AMBER};
        font-weight: 600;
        margin-top: 10px;
    }}

    /* ---- AI narrative card ---- */
    .ai-card {{
        background: linear-gradient(135deg, #FAF7FF 0%, white 60%);
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid #E9DFFB;
        box-shadow: 0 1px 3px rgba(124,58,237,0.08);
        margin-bottom: 14px;
    }}
    .ai-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
        background: {AI_ACCENT}22;
        color: {AI_ACCENT};
    }}
    .ai-text {{
        font-size: 14px;
        color: {NAVY};
        line-height: 1.6;
    }}

    /* ---- No sidebar in this app: hide it and its toggle entirely ---- */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    /* ---- Scenario quick-load bar ---- */
    .scenario-bar-label {{
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: {TEXT_MUTED};
        margin-bottom: 6px;
    }}

    /* ---- Submission form ---- */
    [data-testid="stForm"] {{
        background: white;
        border: 1px solid #EAEEF4;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(11,37,69,0.06);
    }}
    [data-testid="stForm"] .stSlider label,
    [data-testid="stForm"] .stNumberInput label,
    [data-testid="stForm"] .stSelectbox label,
    [data-testid="stForm"] .stCheckbox label {{
        font-size: 12.5px !important;
    }}
    [data-testid="stForm"] [data-testid="stExpander"] summary {{
        font-size: 13.5px;
        font-weight: 700;
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: white;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #EAEEF4;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        color: {TEXT_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ACCENT_SOFT} !important;
        color: {ACCENT} !important;
    }}

    /* ---- Buttons ---- */
    .stFormSubmitButton>button {{
        background: linear-gradient(135deg, {ACCENT} 0%, {NAVY_LIGHT} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        transition: opacity 0.15s ease;
    }}
    .stFormSubmitButton>button:hover {{
        opacity: 0.88;
        color: white;
    }}
    /* Secondary (scenario) buttons stay quiet so the primary action leads */
    .stButton>button {{
        background: white;
        color: {NAVY};
        border: 1px solid #D8DEE9;
        border-radius: 8px;
        font-weight: 600;
        font-size: 12.5px;
        padding: 8px 10px;
        transition: border-color 0.15s ease, background 0.15s ease;
    }}
    .stButton>button:hover {{
        border-color: {ACCENT};
        background: {ACCENT_SOFT};
        color: {NAVY};
    }}

    /* ---- Dataframe ---- */
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #EAEEF4;
    }}

    /* ---- Metric widget cleanup (native st.metric) ---- */
    [data-testid="stMetric"] {{
        background: white;
        border: 1px solid #EAEEF4;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(11,37,69,0.06);
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
        font-weight: 600;
    }}
    [data-testid="stMetricValue"] {{
        color: {NAVY};
    }}
</style>
""", unsafe_allow_html=True)


def _logo_html():
    """Render the company logo if assets/logo.png exists, else a neutral
    fallback glyph. If the image tag itself ever fails to load in the
    browser (bad encoding, deploy artifact stripped, etc.), onerror swaps
    it for the same glyph instead of showing broken-image alt text."""
    if LOGO_PATH.exists():
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        ext = LOGO_PATH.suffix.lstrip(".") or "png"
        return (f'<img src="data:image/{ext};base64,{encoded}" alt="logo" '
                f'onerror="this.replaceWith(Object.assign(document.createElement(\'span\'),'
                f'{{textContent:\'🚘\',style:\'font-size:34px\'}}));"/>')
    return "🚘"


def kpi_card(label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def section_title(text, sub=None):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def ai_card(title, text):
    """A visually distinct card for AI-generated narrative text - kept
    deliberately separate (violet accent, explicit badge) from the
    deterministic rule/model output everywhere else in the app. The AI
    only ever narrates numbers computed elsewhere; it never produces a
    score, band, or premium itself."""
    st.markdown(f"""
        <div class="ai-card">
            <div class="ai-badge">✨ {title}</div>
            <div class="ai-text">{text}</div>
        </div>
    """, unsafe_allow_html=True)


def recommendation_card(alt, base_result):
    color = RECOMMENDATION_COLORS.get(alt.category, ACCENT)
    impact_html = ""
    if alt.result is not None:
        impact_html = f"""
        <div class="rec-impact">
            <span>Composite score: <b>{base_result.composite_score:.1f} &rarr; {alt.result.composite_score:.1f}</b></span>
            <span>Band: <b>{base_result.uw_risk_band} &rarr; {alt.result.uw_risk_band}</b></span>
            <span>Premium: <b>${base_result.final_quoted_premium:,.0f} &rarr; ${alt.result.final_quoted_premium:,.0f}</b></span>
        </div>
        """
    note_html = f'<div class="rec-note">{alt.note}</div>' if alt.note else ""
    st.markdown(f"""
        <div class="rec-card">
            <div class="rec-badge" style="background:{color}22; color:{color};">{alt.category}</div>
            <div class="rec-title">{alt.title}</div>
            <div class="rec-summary">{alt.change_summary}</div>
            {impact_html}
            {note_html}
        </div>
    """, unsafe_allow_html=True)


def driver_bar_chart(reasons, bad_direction, bad_color, good_color, bad_word, good_word):
    """One horizontal bar per top driver, longest first (top to bottom).
    Bars right of the line make the outcome worse, bars left make it
    better, colored accordingly. Horizontal so long text labels get their
    own full-width row - no rotating, no wrapping, no risk of one label
    running into the next, the same fix already applied to Recommended
    Action Mix. The direction word lives in the hover tooltip rather than
    repeated as on-bar text for every single bar."""
    if not reasons:
        return None
    labels, widths, colors, hover_words = [], [], [], []
    for r in reasons:
        bad = r.direction == bad_direction
        labels.append(r.label)
        widths.append(r.weight if bad else -r.weight)
        colors.append(bad_color if bad else good_color)
        hover_words.append(bad_word if bad else good_word)
    fig = go.Figure(go.Bar(
        x=widths, y=labels, orientation="h", marker=dict(color=colors),
        width=0.5,  # pinned thickness: without this, a single bar stretches
                    # to fill the whole category slot (nearly the full chart
                    # height) since there's nothing else to size it against
        customdata=hover_words, hovertemplate="%{y}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        height=min(280, max(140, 70 * len(labels) + 50)), showlegend=False,
        xaxis=dict(range=[-135, 135], zeroline=True, zerolinecolor="#9AA7BD",
                   zerolinewidth=1.5, showticklabels=False, title=None),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        margin=dict(t=10, l=10, r=10, b=10),
    )
    return fig


def appetite_bar_chart(reasons):
    """Appetite reasons are always penalties (points off eligibility), so
    this is simpler than the loss/bind charts: one horizontal bar per
    penalty, all the same color, length = how much it weighs versus the
    others, longest first."""
    if not reasons:
        return None
    fig = go.Figure(go.Bar(
        x=[r.weight for r in reasons], y=[r.label for r in reasons], orientation="h",
        marker=dict(color=SHAP_RED), width=0.5,  # see driver_bar_chart: pins bar thickness
        hovertemplate="%{y}<br>eligibility penalty<extra></extra>",
    ))
    fig.update_layout(
        height=min(280, max(140, 70 * len(reasons) + 50)), showlegend=False,
        xaxis=dict(range=[0, 135], showticklabels=False, title=None),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        margin=dict(t=10, l=10, r=10, b=10),
    )
    return fig


@st.cache_resource
def load_models():
    return get_model_bundle()


@st.cache_data
def load_scored_book():
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def load_market_adj():
    if MARKET_ADJ_PATH.exists():
        return json.loads(MARKET_ADJ_PATH.read_text())
    return {}


@st.cache_data
def load_theft_risk_by_state():
    if THEFT_RISK_PATH.exists():
        return json.loads(THEFT_RISK_PATH.read_text())
    return {}


def compute_vehicle_theft_risk(state: str, anti_theft_available: bool, lookup: dict) -> float:
    """Vehicle theft-risk score is no longer a raw underwriter-entered
    slider - it's derived from the garaging state's historical baseline,
    reduced when an anti-theft device is present. Mirrors how market_adj
    is looked up by state (see load_market_adj)."""
    base = lookup.get(state, 5.5)
    score = base - (ANTI_THEFT_REDUCTION if anti_theft_available else 0.0)
    return round(max(1.0, min(10.0, score)), 1)


def compute_telematics_score(hard_braking_events: float, annual_mileage: float,
                              night_driving_pct: float, speeding_events_month: float) -> float:
    """Telematics safety score, derived from observed driving behavior rather
    than a manually-entered slider. Each penalty term only applies above its
    stated baseline (e.g. mileage under 10,000/yr or night driving under 15%
    contributes nothing). Median-of-3 against [0, 100] clips the result into
    range - the same pattern rules_engine.apply_premium_guardrails uses for
    the renewal premium band."""
    penalty = (
        1.5 * hard_braking_events
        + 3 * speeding_events_month
        + max(0.0, 0.8 * (night_driving_pct - 15))
        + max(0.0, 0.5 * (annual_mileage - 10_000) / 1_000)
    )
    return sorted([0.0, 100.0, 100 - penalty])[1]


bundle = load_models()
book = load_scored_book()
market_adj_lookup = load_market_adj()
theft_risk_lookup = load_theft_risk_by_state()

# ============================================================================
# BRAND / HEADER BAR  (logo top-left)
# ============================================================================
st.markdown(f"""
<div class="brand-bar">
    <div class="brand-logo">{_logo_html()}</div>
    <div class="brand-text">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
</div>
""", unsafe_allow_html=True)

tab_dashboard, tab_interface = st.tabs(["Dashboard", "Interface"])

# ============================================================================
# DASHBOARD TAB
# ============================================================================
with tab_dashboard:
    section_title("Filters")
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        states = ["All States"] + sorted(book["State"].dropna().unique().tolist())
        f_state = c1.selectbox("State", states)
        bands = ["All Risk Bands"] + sorted(book["uw_risk_band"].dropna().unique().tolist())
        f_band = c2.selectbox("Risk Band", bands)
        channels = ["All Channels"] + sorted(book["Submission_Channel"].dropna().unique().tolist())
        f_channel = c3.selectbox("Channel", channels)
        nr = ["All", "New Business", "Renewal"]
        f_nr = c4.selectbox("New/Renewal", nr)
        body_types = ["All Body Types"] + sorted(book["Body_Type"].dropna().unique().tolist())
        f_body = c5.selectbox("Body Type", body_types)

    filtered = book.copy()
    if f_state != "All States":
        filtered = filtered[filtered["State"] == f_state]
    if f_band != "All Risk Bands":
        filtered = filtered[filtered["uw_risk_band"] == f_band]
    if f_channel != "All Channels":
        filtered = filtered[filtered["Submission_Channel"] == f_channel]
    if f_nr != "All":
        filtered = filtered[filtered["New_Renewal_Label"] == f_nr]
    if f_body != "All Body Types":
        filtered = filtered[filtered["Body_Type"] == f_body]

    section_title("Portfolio Overview")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        kpi_card("Submissions", f"{len(filtered):,}")
    with k2:
        kpi_card("Avg Composite Score", f"{filtered['composite_score'].mean():.1f}" if len(filtered) else "–")
    with k3:
        stp_rate = (filtered["recommended_action"] == "Auto-Quote / STP").mean() if len(filtered) else 0
        kpi_card("STP Rate", f"{stp_rate:.1%}")
    with k4:
        decline_rate = (filtered["uw_risk_band"] == "Decline").mean() if len(filtered) else 0
        kpi_card("Decline Rate", f"{decline_rate:.1%}")
    with k5:
        kpi_card("Avg Quoted Premium", f"${filtered['final_quoted_premium'].mean():,.0f}" if len(filtered) else "–")
    with k6:
        kpi_card("Avg Loss Ratio", f"{filtered['expected_loss_ratio'].mean():.1%}" if len(filtered) else "–",
                  sub="current-rate basis")

    st.write("")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        section_title("Risk Band Distribution")
        band_counts = filtered["uw_risk_band"].value_counts().reset_index()
        band_counts.columns = ["Band", "Count"]
        order = [b for b in ["Preferred", "Standard", "Non-Standard", "Decline", "Hard Stop"] if b in band_counts["Band"].values]
        fig = px.bar(
            band_counts, x="Band", y="Count", color="Band",
            category_orders={"Band": order},
            color_discrete_map=BAND_COLORS,
            text="Count",
        )
        fig.update_traces(textposition="outside", marker_line_width=0,
                          hovertemplate="%{x}: %{y:,}<extra></extra>")
        fig.update_layout(showlegend=False, height=360, xaxis_title=None, yaxis_title="Submissions")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        section_title("Recommended Action Mix")
        action_counts = filtered["recommended_action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        action_counts["Action"] = action_counts["Action"].map(ACTION_SHORT_LABELS).fillna(action_counts["Action"])
        total = action_counts["Count"].sum()
        action_counts["Pct"] = action_counts["Count"] / total if total else 0
        action_counts = action_counts.sort_values("Count", ascending=True)
        fig2 = px.bar(
            action_counts, x="Count", y="Action", orientation="h",
            color="Action", color_discrete_map=ACTION_COLORS,
            text=[f"{c:,}  ({p:.1%})" for c, p in zip(action_counts["Count"], action_counts["Pct"])],
        )
        fig2.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
        fig2.update_layout(
            showlegend=False, height=360, xaxis_title="Submissions", yaxis_title=None,
            margin=dict(l=10, r=90, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        section_title("Avg Composite Score by State")
        by_state = filtered.groupby("State")["composite_score"].mean().round(1).reset_index().sort_values(
            "composite_score", ascending=False)
        fig3 = px.bar(by_state, x="State", y="composite_score", color="composite_score",
                       color_continuous_scale=[ACCENT_SOFT, ACCENT, NAVY])
        fig3.update_traces(hovertemplate="%{x}: %{y:.1f}<extra></extra>")
        fig3.update_layout(height=380, xaxis_title=None, yaxis_title="Avg Composite Score",
                            yaxis_tickformat=".0f", coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        section_title("Technical vs. Final Quoted Premium (sample)")
        sample = filtered.sample(min(500, len(filtered)), random_state=1) if len(filtered) else filtered
        fig4 = px.scatter(
            sample, x="technical_premium", y="final_quoted_premium",
            color="uw_risk_band", opacity=0.7,
            color_discrete_map=BAND_COLORS,
            labels={"technical_premium": "Technical Premium ($)",
                    "final_quoted_premium": "Final Quoted Premium ($)"},
        )
        fig4.add_shape(type="line", x0=0, y0=0, x1=16000, y1=16000,
                        line=dict(dash="dash", color=TEXT_MUTED, width=1.5))
        fig4.update_traces(hovertemplate="Technical: $%{x:,.0f}<br>Final: $%{y:,.0f}")
        fig4.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=-0.35))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    section_title("Rate Adequacy Exceptions")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    exceptions = filtered[filtered["rate_adequacy_flag"] == True]
    pct = len(exceptions) / max(len(filtered), 1)
    st.markdown(
        f'<p style="color:{TEXT_MUTED}; font-size:13px; margin-top:-4px;">'
        f'<b style="color:{NAVY}">{len(exceptions):,}</b> of {len(filtered):,} submissions '
        f'(<b style="color:{RED if pct > 0.1 else NAVY}">{pct:.1%}</b>) flagged: final quoted premium below technical premium</p>',
        unsafe_allow_html=True,
    )
    show_cols = ["Policy_ID", "State", "uw_risk_band", "technical_premium", "final_quoted_premium",
                 "composite_score", "recommended_action"]
    display_df = exceptions[show_cols].head(50).copy()
    display_df["technical_premium"] = display_df["technical_premium"].round(0)
    display_df["final_quoted_premium"] = display_df["final_quoted_premium"].round(0)
    display_df["composite_score"] = display_df["composite_score"].round(1)
    st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        column_config={
            "Policy_ID": st.column_config.TextColumn("Policy ID"),
            "uw_risk_band": st.column_config.TextColumn("Risk Band"),
            "technical_premium": st.column_config.NumberColumn("Technical Premium", format="$%.0f"),
            "final_quoted_premium": st.column_config.NumberColumn("Final Quoted Premium", format="$%.0f"),
            "composite_score": st.column_config.NumberColumn("Composite Score", format="%.1f"),
            "recommended_action": st.column_config.TextColumn("Recommended Action"),
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# INTERFACE TAB (live risk scoring, suggestions, explainability)
# ============================================================================
with tab_interface:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{NAVY} 0%,{NAVY_LIGHT} 100%);
                border-radius:14px;padding:18px 24px;margin-bottom:16px;
                box-shadow:0 4px 18px rgba(11,37,69,0.18);">
        <div style="color:#C6D4EA;font-size:11.5px;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:4px;">Single Submission</div>
        <div style="color:white;font-size:21px;font-weight:800;line-height:1.2;">UW Score What-If Simulator</div>
        <div style="color:#C6D4EA;font-size:13px;margin-top:5px;">
            Score one risk end to end: eligibility, pricing, coverage fit, explainability, and underwriter alternatives.
        </div>
    </div>
    """, unsafe_allow_html=True)

    def _iv(key, fallback):
        """Current value for an input, from session state if a scenario set it."""
        return st.session_state.get(f"in_{key}", fallback)

    def _cb(container, label, field, default=False, **kwargs):
        """Checkbox bound to session_state via an explicit key. Unlike
        selectbox/slider/number_input, st.checkbox does not reliably pick up
        a changed `value=` on a later rerun once the user has ever toggled
        it - the prior interaction sticks regardless of a new default. Every
        checkbox that needs to respond to a loaded scenario or Reset has to
        go through session_state via an explicit key instead."""
        st.session_state.setdefault(f"in_{field}", default)
        return container.checkbox(label, key=f"in_{field}", **kwargs)

    SCENARIO_PLACEHOLDER = "— Select a scenario (optional) —"

    def _load_scenario():
        chosen = st.session_state.get("scenario_select")
        if not chosen or chosen == SCENARIO_PLACEHOLDER:
            return
        meta = SCENARIOS[chosen]
        for k, v in meta["inputs"].items():
            st.session_state[f"in_{k}"] = v
        # Annual mileage is a keyed dropdown of 1000-multiples (max 100,000):
        # its stored value must always be a valid option, so a scenario's
        # exact historical mileage (e.g. 6,393) gets snapped to the nearest
        # multiple of 1,000 rather than rejected by the widget.
        if "annual_mileage" in meta["inputs"]:
            _snap_options = list(range(0, 100_001, 1000))
            st.session_state["in_annual_mileage"] = min(
                _snap_options, key=lambda o: abs(o - meta["inputs"]["annual_mileage"])
            )
        # New/renewal is now driven by the Existing Customer checkbox above
        # the form rather than a selectbox inside it.
        st.session_state["in_existing_customer"] = meta["inputs"].get("new_renewal") == "Renewal"
        st.session_state["live_margin_pct"] = meta["inputs"].get("target_margin_pct", 5)
        st.session_state["has_result"] = True

    def _reset_scenario():
        """Reset clears the form back to a blank/zero state (not a demo
        scenario's values) - every field in RESET_DEFAULTS goes to its
        zero/off/most-neutral value."""
        for k, v in RESET_DEFAULTS.items():
            st.session_state[f"in_{k}"] = v
        st.session_state["live_margin_pct"] = 5
        st.session_state["has_result"] = False
        st.session_state["scenario_select"] = SCENARIO_PLACEHOLDER

    # ---- Reset sits above/outside the scenario picker so it always works,
    # even with the picker collapsed - it clears the whole submission, not
    # just whatever scenario happens to be loaded.
    _, reset_col = st.columns([4, 1])
    reset_col.button("Reset", use_container_width=True, on_click=_reset_scenario,
                      help="Clear every section back to its blank/default value")

    # ---- Pre-built scenarios - collapsed by default; selecting one loads
    # and scores it immediately, leaving it on the placeholder just goes
    # straight to the sections below with nothing pre-filled. -------------
    with st.expander("Load a scenario"):
        st.selectbox(
            "Scenario", [SCENARIO_PLACEHOLDER] + SCENARIO_NAMES, key="scenario_select",
            on_change=_load_scenario, label_visibility="collapsed",
        )
        _chosen_scenario = st.session_state.get("scenario_select")
        if _chosen_scenario and _chosen_scenario != SCENARIO_PLACEHOLDER:
            _meta = SCENARIOS[_chosen_scenario]
            st.caption(f"{_meta['blurb']}  (expects: {_meta['expect']})")

    st.write("")

    # ---- Existing customer / coverage - submission-level, above the
    # numbered sections ---------------------------------------------------
    top1, top2 = st.columns(2)
    with top1:
        st.markdown('<div class="scenario-bar-label">Existing Customer</div>', unsafe_allow_html=True)
        st.session_state.setdefault("in_existing_customer", False)
        existing_customer = st.checkbox(
            "Existing customer?", key="in_existing_customer",
            help="Checked = this is a renewal of a current policy. Unchecked = new business.",
        )
    with top2:
        st.markdown('<div class="scenario-bar-label">Current / Required Coverage</div>', unsafe_allow_html=True)
        coverage_type = st.selectbox(
            "Coverage", COVERAGE_OPTIONS, index=COVERAGE_OPTIONS.index(_iv("coverage_type", DEFAULT_COVERAGE)),
            key="in_coverage_type", label_visibility="collapsed",
        )

    st.write("")

    # ---- Input sections, stacked top to bottom (1 open by default) ------
    # Sections 1-3 live outside the form: Telematics needs the safety-score
    # slider to appear the instant "Telematics enrolled" is checked, and
    # st.form batches every widget inside it until Calculate is pressed - no
    # widget inside a form can react to another widget's change before then.
    # Sections 4-6 stay batched in the form below since none of them need
    # that live behaviour, and batching keeps their sliders from re-scoring
    # the submission on every drag once a result is already showing.
    # Sections 1-3 live outside st.form: Telematics needs its driving-behavior
    # dropdowns to appear the instant "Telematics enrolled" is checked, and
    # st.form batches every widget until Calculate is pressed - no widget
    # inside a form can react to another widget's change before then. Claims
    # History has no such need but sits between Driver & Loss Risk and
    # Telematics in the required section order, so it moves outside the form
    # too rather than breaking that order. Only Appetite & Eligibility and
    # Bind & Market stay batched inside the form below.
    _states = sorted(book["State"].dropna().unique().tolist())

    with st.expander("1. Driver & Loss Risk Parameters", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        driver_age = c1.number_input("Driver age (years)", 16, 99, _iv("driver_age", 16), key="in_driver_age")
        if driver_age > MAX_TRAINED_DRIVER_AGE:
            c1.markdown(
                f'<div style="color:{AMBER}; font-size:11.5px; margin-top:-8px;">'
                f'Beyond the oldest driver in the training data ({MAX_TRAINED_DRIVER_AGE}). '
                f'The models are extrapolating here, not applying a calibrated pattern.</div>',
                unsafe_allow_html=True,
            )
        # US license eligibility is 16, so years licensed can never exceed
        # driver age minus 16 (e.g. age 32 -> at most 16 years licensed).
        # Clamped directly in session_state (not just the initial default)
        # since it's now a keyed widget: if the user raises years_licensed
        # near the old max and then lowers driver_age, the stored value must
        # be pulled back in range before the widget re-renders or Streamlit
        # raises a bounds error instead of just re-clamping silently.
        max_years_licensed = max(0, int(driver_age) - 16)
        st.session_state["in_years_licensed"] = min(_iv("years_licensed", 0), max_years_licensed)
        years_licensed = c2.number_input("Years licensed", 0, max_years_licensed, key="in_years_licensed")
        _mileage_options = list(range(0, 100_001, 1000))
        _mileage_default = min(_mileage_options, key=lambda o: abs(o - _iv("annual_mileage", 0)))
        st.session_state["in_annual_mileage"] = _mileage_default
        annual_mileage = c3.selectbox("Annual mileage", _mileage_options,
                                       key="in_annual_mileage", format_func=lambda v: f"{v:,}")
        _state_default = _iv("state", "TX")
        if _state_default not in _states:
            _state_default = _states[0]
        st.session_state["in_state"] = _state_default
        state = c4.selectbox("Garaging state", _states, key="in_state")
        territory_type = c5.selectbox("Territory type", ["Suburban", "Urban", "Rural"],
                                       index=["Suburban", "Urban", "Rural"].index(_iv("territory_type", "Suburban")),
                                       key="in_territory_type")

        c1, c2, c3, c4, c5 = st.columns(5)
        vehicle_use = c1.selectbox("Vehicle use", ["Pleasure", "Commute", "Business"],
                                    index=["Pleasure", "Commute", "Business"].index(_iv("vehicle_use", "Pleasure")),
                                    key="in_vehicle_use")
        body_type = c2.selectbox("Vehicle type", BODY_TYPE_OPTIONS,
                                  index=BODY_TYPE_OPTIONS.index(_iv("body_type", "Sedan")), key="in_body_type")
        coverage_lapse_days = c3.number_input("Coverage lapse (days)", 0, 365, _iv("coverage_lapse_days", 0),
                                               key="in_coverage_lapse_days")
        vehicle_value = c4.number_input("Vehicle value ($)", 500, 200000, _iv("vehicle_value", 500), step=500,
                                         key="in_vehicle_value")
        anti_theft_available = _cb(c5, "Anti-theft device available?", "anti_theft_available", False)

        c1, c2, c3, c4 = st.columns(4)
        collision_deductible = c1.number_input("Collision deductible ($)", 0, 5000,
                                                _iv("collision_deductible", 0), step=100,
                                                key="in_collision_deductible")
        comprehensive_deductible = c2.number_input("Comprehensive deductible ($)", 0, 5000,
                                                    _iv("comprehensive_deductible", 0), step=100,
                                                    key="in_comprehensive_deductible")
        repairability_score = c3.slider("Repairability score (0-10)", 0, 10, _iv("repairability_score", 0),
                                         key="in_repairability_score")
        parts_cost_index = c4.number_input("Parts cost index", 0.5, 2.0, float(_iv("parts_cost_index", 0.5)),
                                            step=0.05, key="in_parts_cost_index")

        c1, c2 = st.columns(2)
        undeclared_business_use = _cb(c1, "Undeclared business use?", "undeclared_business_use", False)
        luxury_vehicle = _cb(c2, "Luxury vehicle?", "luxury_vehicle", False)

        _factors = get_state_factors(state)
        labor_cost_index = _factors["labor_cost_index"]
        medical_cost_index = _factors["medical_cost_index"]
        litigation_environment = _factors["litigation_environment"]
        st.caption(
            f"Auto-populated from {state}: Labor Cost Index **{labor_cost_index:.2f}**, "
            f"Medical Cost Index **{medical_cost_index:.2f}**, Liability Environment "
            f"**{litigation_environment:.2f}**. Vehicle theft-risk is also calculated automatically "
            f"from the garaging state and anti-theft device availability."
        )

    with st.expander("2. Claims History"):
        c1, c2, c3, c4, c5 = st.columns(5)
        prior_claims_3y = c1.number_input("Prior claims (3yr)", 0, 20, _iv("prior_claims_3y", 0),
                                           key="in_prior_claims_3y")
        at_fault_claims_3y = c2.number_input("At-fault claims (3yr)", 0, 20, _iv("at_fault_claims_3y", 0),
                                              key="in_at_fault_claims_3y")
        prior_claims_5y = c3.number_input("Prior claims (5yr)", 0, 20, _iv("prior_claims_5y", 0),
                                           key="in_prior_claims_5y")
        moving_violations_3y = c4.number_input("Moving violations (3yr)", 0, 20, _iv("moving_violations_3y", 0),
                                                key="in_moving_violations_3y")
        speeding_violations_3y = c5.number_input("Speeding violations (3yr)", 0, 20,
                                                  _iv("speeding_violations_3y", 0), key="in_speeding_violations_3y")
        c1, c2, c3 = st.columns(3)
        major_violation = _cb(c1, "Major violation?", "major_violation", False)
        dui_flag = _cb(c2, "DUI on record?", "dui_flag", False)
        license_suspension = _cb(c3, "License suspension?", "license_suspension", False)

    with st.expander("3. Telematics", expanded=True):
        telematics_enrolled = _cb(st, "Telematics enrolled?", "telematics_enrolled", False)
        if telematics_enrolled:
            c1, c2 = st.columns(2)
            _hb_options = list(range(0, 41))
            hard_braking_events = c1.selectbox(
                "Hard-braking events / month", _hb_options,
                index=_hb_options.index(min(_hb_options, key=lambda o: abs(o - _iv("hard_braking_events", 0)))),
                key="in_hard_braking_events",
            )
            _sp_options = list(range(0, 31))
            speeding_events_month = c2.selectbox(
                "Speeding events / month", _sp_options,
                index=_sp_options.index(min(_sp_options, key=lambda o: abs(o - _iv("speeding_events_month", 0)))),
                key="in_speeding_events_month",
            )
            _nd_options = [0, 5, 10, 15, 20]
            night_driving_pct = st.selectbox(
                "Night driving (%)", _nd_options,
                index=_nd_options.index(min(_nd_options, key=lambda o: abs(o - _iv("night_driving_pct", 0)))),
                key="in_night_driving_pct",
            )
            telematics_safety_score = compute_telematics_score(
                hard_braking_events, annual_mileage, night_driving_pct, speeding_events_month
            )
            st.metric("Telematics safety score (computed)", f"{telematics_safety_score:.0f}")
            st.caption("Computed from the driving-behavior inputs above using the standard telematics "
                       "scoring formula - not a manual entry.")
        else:
            hard_braking_events = _iv("hard_braking_events", 0)
            night_driving_pct = _iv("night_driving_pct", 0)
            speeding_events_month = _iv("speeding_events_month", 0)
            telematics_safety_score = compute_telematics_score(
                hard_braking_events, annual_mileage, night_driving_pct, speeding_events_month
            )
            st.caption("Check Telematics enrolled to enter driving-behavior details and compute a "
                       "safety score. A score can't exist for a driver who was never measured, so "
                       "it's ignored (imputed) otherwise.")

    with st.form("interface_form"):
        with st.expander("4. Appetite & Eligibility"):
            c1, c2, c3, c4 = st.columns(4)
            cat_exposure_score = c1.slider("CAT exposure score (0-10)", 0, 10, _iv("cat_exposure_score", 0),
                                            key="in_cat_exposure_score")
            state_in_appetite = _cb(c2, "Garaging state in footprint?", "state_in_appetite", False)
            prior_total_loss = _cb(c3, "Prior total loss?", "prior_total_loss", False)
            confirmed_fraud = _cb(c4, "Fraud indicator?", "confirmed_fraud", False)

        with st.expander("5. Bind & Market"):
            c1, c2, c3 = st.columns(3)
            customer_tenure_years = c1.number_input("Customer tenure (years)", 0, 50,
                                                     _iv("customer_tenure_years", 0),
                                                     key="in_customer_tenure_years")
            multi_policy = _cb(c2, "Multi-policy eligible?", "multi_policy", False)
            submission_channel = c3.selectbox("Channel", ["Direct", "Agent", "Aggregator"],
                                               index=["Direct", "Agent", "Aggregator"].index(_iv("submission_channel", "Direct")),
                                               key="in_submission_channel")
            c1, c2, c3 = st.columns(3)
            quote_revisions = c1.number_input("Quote revisions", 0, 20, _iv("quote_revisions", 0),
                                               key="in_quote_revisions")
            price_sensitivity = c2.selectbox("Price sensitivity", ["Low", "Medium", "High"],
                                              index=["Low", "Medium", "High"].index(_iv("price_sensitivity", "Low")),
                                              key="in_price_sensitivity")
            competitor_premium_estimate = c3.number_input("Competitor premium est. ($, optional)", 0, 30000,
                                                           _iv("competitor_premium_estimate", 0), step=100,
                                                           key="in_competitor_premium_estimate")
            prior_year_premium = st.number_input("Prior year premium ($, renewals)", 0, 30000,
                                                  _iv("prior_year_premium", 0), step=100,
                                                  key="in_prior_year_premium")
            st.caption("Only used when Existing Customer is checked above.")

        st.write("")
        submitted = st.form_submit_button("Calculate Risk Score", type="primary", use_container_width=True)

    if submitted:
        st.session_state["has_result"] = True
    show_results = st.session_state.get("has_result", False)

    if not show_results:
        st.markdown(f"""
        <div style="background:white;border:1px dashed {ACCENT}55;border-radius:14px;
                    padding:44px 32px;text-align:center;margin-top:8px;">
            <div style="font-size:15px;font-weight:700;color:{NAVY};margin-bottom:6px;">
                Load a scenario above, or set the inputs and click
                <span style="color:{ACCENT}">Calculate Risk Score</span>.
            </div>
            <div style="color:{TEXT_MUTED};font-size:13.5px;max-width:660px;margin:0 auto;">
                You'll get the underwriting verdict and price, a coverage-fit comparison across every
                coverage form, a Shapley breakdown of what drove the score, and, where the answer would
                otherwise be a decline, the alternatives an underwriter should consider first.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if show_results:
        new_renewal = "Renewal" if existing_customer else "New Business"
        computed_theft_risk = compute_vehicle_theft_risk(state, anti_theft_available, theft_risk_lookup)
        # Read before the slider below is instantiated: Streamlit updates
        # session_state for a keyed widget before the script body reruns, so
        # this already reflects the latest drag even on a rerun the slider
        # itself triggered - the same pattern _iv() uses for scenario loads.
        # (setdefault, not value=, on the widget itself - a keyed widget's
        # value can't be set both ways without Streamlit warning about it.)
        st.session_state.setdefault("live_margin_pct", 5)
        current_margin_pct = st.session_state["live_margin_pct"]

        submission = SubmissionInputs(
            driver_age=driver_age, years_licensed=years_licensed, annual_mileage=annual_mileage,
            vehicle_use=vehicle_use, territory_type=territory_type,
            # No longer collected by the form - imputed to the training
            # median rather than guessed, same convention as every other
            # UI-uncollected field (see row_mapper._MODEL_ROW_DEFAULTS).
            traffic_congestion=float("nan"), vehicle_theft_risk=computed_theft_risk,
            coverage_lapse_days=coverage_lapse_days, undeclared_business_use=undeclared_business_use,
            prior_claims_3y=prior_claims_3y, at_fault_claims_3y=at_fault_claims_3y,
            prior_claims_5y=prior_claims_5y, moving_violations_3y=moving_violations_3y,
            speeding_violations_3y=speeding_violations_3y, major_violation=major_violation,
            dui_flag=dui_flag, license_suspension=license_suspension,
            telematics_enrolled=telematics_enrolled, telematics_safety_score=telematics_safety_score,
            vehicle_value=vehicle_value, repairability_score=repairability_score,
            parts_cost_index=parts_cost_index, labor_cost_index=labor_cost_index,
            medical_cost_index=medical_cost_index, litigation_environment=litigation_environment,
            # No longer collected by the form - assumed non-performance.
            luxury_vehicle=luxury_vehicle, performance_vehicle=False,
            collision_deductible=collision_deductible, comprehensive_deductible=comprehensive_deductible,
            cat_exposure_score=cat_exposure_score, state_in_appetite=state_in_appetite,
            prior_total_loss=prior_total_loss, confirmed_fraud=confirmed_fraud,
            new_renewal=new_renewal, customer_tenure_years=customer_tenure_years,
            multi_policy=multi_policy, submission_channel=submission_channel, state=state,
            quote_revisions=quote_revisions, competing_quotes_count=0,
            price_sensitivity=price_sensitivity,
            competitor_premium_estimate=competitor_premium_estimate if competitor_premium_estimate > 0 else None,
            prior_year_premium=prior_year_premium if new_renewal == "Renewal" else None,
            coverage_type=coverage_type,
            target_profit_margin=current_margin_pct / 100,
            anti_theft_available=anti_theft_available,
            body_type=body_type,
        )

        result = score_live_submission(bundle, submission, book, market_adj_lookup)
        explanation = explain_submission(bundle, submission, result)

        st.divider()
        band_color = BAND_COLORS.get(result.uw_risk_band, NAVY)
        st.markdown(f"""
        <div class="result-banner" style="background: linear-gradient(135deg, {band_color} 0%, {NAVY} 140%);">
            <div style="font-size:13px; opacity:0.85; text-transform:uppercase; letter-spacing:0.05em;">Result</div>
            <div style="font-size:26px; font-weight:800; margin-top:4px;"><span class="band-dot"></span>{result.uw_risk_band}</div>
            <div style="font-size:14px; margin-top:6px; opacity:0.95;">Recommended Action: <b>{result.recommended_action}</b></div>
            <div style="font-size:13px; margin-top:10px; opacity:0.9;">{explanation.summary}</div>
        </div>
        """, unsafe_allow_html=True)

        if result.hard_stop:
            st.error("Hard stop triggered: " + " | ".join(result.hard_stop_reasons))

        # -----------------------------------------------------------------
        # Technical Premium & Profit Margin - the margin slider is live:
        # moving it re-scores immediately, no need to click Calculate again.
        # -----------------------------------------------------------------
        section_title("Technical Premium & Pricing",
                      sub="Move the profit margin slider to see the technical and final premium update live.")
        col_base, col_tp, col_slider, col_fp = st.columns([1, 1, 1.3, 1])
        with col_base:
            st.metric("Base Loss Cost", f"${result.expected_loss_load:,.0f}",
                      help="Expected Loss Cost (frequency x severity), trended and developed: "
                           "frequency x severity x 1.06 (trend) x 1.03 (development).")
        with col_tp:
            st.metric("Technical Premium", f"${result.technical_premium:,.0f}",
                      help=(f"(Base Loss Cost x (1 + LAE 10% + Expense 18% + CAT load {result.cat_load:.1%} "
                            f"+ Risk load {result.risk_load:.1%})) / (1 - Profit Margin {current_margin_pct}%)"))
        with col_slider:
            st.select_slider("Profit margin (%)", options=[2, 3, 4, 5, 6, 7], key="live_margin_pct")
        with col_fp:
            st.metric("Final Quoted Premium", f"${result.final_quoted_premium:,.0f}",
                      delta=f"{(result.final_quoted_premium/result.technical_premium - 1):+.1%} vs technical")
        if result.rate_adequacy_flag:
            st.warning("Rate Adequacy Exception: final quoted premium is below technical premium.")

        if ai_narrator.is_available():
            with st.spinner("Generating AI underwriting narrative..."):
                ai_summary = ai_narrator.narrate_decision(result, explanation)
            if ai_summary:
                ai_card("AI Underwriting Narrative", ai_summary)

        # -----------------------------------------------------------------
        # Coverage Fit - same risk priced under every coverage form
        # -----------------------------------------------------------------
        section_title(
            "Suggested Coverage and Coverage Fit & Suggestion",
            sub="Coverage opted by the customer, compared with the coverage this risk is best suited "
                "for. Coverage changes the flags the loss models read, so each option carries its own "
                "technical premium, not just an appetite adjustment.",
        )
        with st.spinner("Pricing every coverage option..."):
            cov = coverage_fit_analysis(bundle, submission, book, market_adj_lookup)

        col_current, col_suggested = st.columns(2)
        with col_current:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f'<div style="font-weight:700;color:{NAVY};margin-bottom:8px;">'
                        f'Coverage Opted by Customer</div>', unsafe_allow_html=True)
            st.markdown(f"**{cov.current.coverage_type}**")
            m1, m2 = st.columns(2)
            m1.metric("Band", cov.current.uw_risk_band)
            m2.metric("Technical Premium", f"${cov.current.technical_premium:,.0f}")
            st.metric("Final Quoted Premium", f"${cov.current.final_quoted_premium:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_suggested:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            same_note = "  _(same as current)_" if cov.best.coverage_type == cov.current.coverage_type else ""
            st.markdown(f'<div style="font-weight:700;color:{NAVY};margin-bottom:8px;">'
                        f'Suggested Coverage</div>', unsafe_allow_html=True)
            st.markdown(f"**{cov.best.coverage_type}**{same_note}")
            m1, m2 = st.columns(2)
            m1.metric("Band", cov.best.uw_risk_band)
            m2.metric("Technical Premium", f"${cov.best.technical_premium:,.0f}")
            st.metric("Final Quoted Premium", f"${cov.best.final_quoted_premium:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Renewal guardrail transparency: if every option quotes to the same
        # premium, it's the +/-90%-vs-last-year renewal band clamping all of
        # them identically, not a computation error - worth saying so
        # explicitly rather than leaving that to look like a bug.
        if submission.new_renewal == "Renewal" and submission.prior_year_premium:
            lo = submission.prior_year_premium * RENEWAL_BAND_LOW
            hi = submission.prior_year_premium * RENEWAL_BAND_HIGH
            quoted_values = {round(f.final_quoted_premium) for f in cov.options}
            if len(quoted_values) == 1:
                st.caption(
                    f"All five options quote to the same ${next(iter(quoted_values)):,.0f}: this is a "
                    f"renewal, so pricing is held between ${lo:,.0f} and ${hi:,.0f} "
                    f"({RENEWAL_BAND_LOW:.0%}-{RENEWAL_BAND_HIGH:.0%} of last year's ${submission.prior_year_premium:,.0f}) "
                    f"regardless of coverage, and every option's underlying premium falls outside that band "
                    f"on the same side."
                )
            else:
                st.caption(
                    f"Renewal guardrail in effect: quoted premium is held between ${lo:,.0f} and ${hi:,.0f} "
                    f"({RENEWAL_BAND_LOW:.0%}-{RENEWAL_BAND_HIGH:.0%} of last year's ${submission.prior_year_premium:,.0f})."
                )

        rec_color = GREEN if cov.improves else ACCENT
        st.markdown(f"""
            <div style="background:{rec_color}12;border-left:4px solid {rec_color};
                        border-radius:8px;padding:14px 16px;margin:4px 0 18px 0;">
                <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.05em;color:{rec_color};margin-bottom:5px;">
                    {'Suggested coverage' if cov.improves else 'Coverage assessment'}
                </div>
                <div style="font-size:14px;color:{NAVY};line-height:1.55;">{cov.message}</div>
            </div>
        """, unsafe_allow_html=True)

        # -----------------------------------------------------------------
        # Key Factors Behind This Decision - a quick chip digest, collapsed
        # until the underwriter opens it. The deeper numeric/chart breakdown
        # lives in its own "Why This Score" expander below, same collapsed
        # pattern as Technical & Audit detail. Color convention throughout:
        # blue = helps the outcome, red = hurts it - kept consistent across
        # all three charts (appetite reasons are always penalties, so that
        # chart is red-only).
        # -----------------------------------------------------------------
        with st.expander("Key Factors Behind This Decision"):
            if explanation.top_reasons:
                chips = "".join(
                    f'<span style="display:inline-block; background:{ACCENT_SOFT}; color:{NAVY}; '
                    f'font-size:12px; font-weight:600; padding:5px 12px; border-radius:20px; '
                    f'margin:0 6px 6px 0;">{rc.label}</span>'
                    for rc in explanation.top_reasons
                )
                st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
            else:
                st.caption("No material factors to display.")

        with st.expander("Why This Score"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Loss Propensity Score", f"{result.loss_propensity_score:.1f}")
            m2.metric("Bind Propensity Score", f"{result.bind_propensity_score:.1f}")
            m3.metric("Appetite Score", f"{result.appetite_score:.1f}")

            m1, m2 = st.columns(2)
            m1.metric("Decision Confidence", explanation.confidence)
            m2.metric("Manual Review Required", "Yes" if explanation.manual_review else "No")

            m1, m2 = st.columns(2)
            m1.metric("Predicted Claim Frequency", f"{result.expected_claim_frequency:.3f}")
            m2.metric("Predicted Severity", f"${result.expected_severity:,.0f}")

            st.caption("Top factors behind each score - standard SHAP coloring: "
                       "blue bars help the outcome, red bars hurt it.")
            col_l, col_bd, col_ap = st.columns(3)
            with col_l:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(f'<div style="font-weight:700; color:{NAVY}; margin-bottom:4px;">Loss Risk Drivers</div>', unsafe_allow_html=True)
                fig = driver_bar_chart(explanation.loss_reasons, bad_direction="increases",
                                        bad_color=SHAP_RED, good_color=SHAP_BLUE,
                                        bad_word="raises loss cost", good_word="lowers loss cost")
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption("No material loss-risk drivers.")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_bd:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(f'<div style="font-weight:700; color:{NAVY}; margin-bottom:4px;">Bind Propensity Drivers</div>', unsafe_allow_html=True)
                fig = driver_bar_chart(explanation.bind_reasons, bad_direction="decreases",
                                        bad_color=SHAP_RED, good_color=SHAP_BLUE,
                                        bad_word="lowers bind likelihood", good_word="raises bind likelihood")
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption("No material bind-propensity drivers.")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_ap:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(f'<div style="font-weight:700; color:{NAVY}; margin-bottom:4px;">Appetite / Eligibility Drivers</div>', unsafe_allow_html=True)
                fig = appetite_bar_chart(explanation.appetite_reasons)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption("No eligibility penalties. Clean appetite profile.")
                st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------------------------------------------
        # Underwriter Recommendations - alternatives to a flat decline
        # -----------------------------------------------------------------
        if result.uw_risk_band in ("Decline", "Hard Stop"):
            section_title(
                "Underwriter Recommendations",
                sub="Alternatives to a flat decline, re-scored through the same rules engine.",
            )
            alternatives = build_alternatives(bundle, submission, result, book, market_adj_lookup)
            if not alternatives:
                st.info(
                    "No underwriting alternative available. Every rule triggered here is a "
                    "mandatory decline (DUI/DWI, license suspension, or confirmed fraud) with no "
                    "discretionary override."
                )
            else:
                if ai_narrator.is_available():
                    with st.spinner("Generating AI recommendation narrative..."):
                        ai_rec_summary = ai_narrator.narrate_alternatives(result, alternatives)
                    if ai_rec_summary:
                        ai_card("AI Recommendation Summary", ai_rec_summary)
                for alt in alternatives:
                    recommendation_card(alt, result)

        with st.expander("Technical & audit detail"):
            st.caption("Every figure below is deterministic and reproducible. The same inputs "
                       "always produce the same output. The AI narrative above only explains "
                       "these numbers in plain English; it never alters them.")
            tab_calc, tab_reasons = st.tabs(["Full calculation", "Reason codes"])
            with tab_calc:
                # Stringify every value - result.__dict__ mixes bool/float/str,
                # and a single object-dtype column with mixed Python types
                # breaks pandas' Arrow serialization in st.dataframe.
                calc_rows = [
                    {"Field": k.replace("_", " ").title(),
                     "Value": str(round(v, 3) if isinstance(v, float) else v)}
                    for k, v in result.__dict__.items() if k != "hard_stop_reasons"
                ]
                st.dataframe(pd.DataFrame(calc_rows), use_container_width=True, hide_index=True)
            with tab_reasons:
                reason_rows = [
                    {"Category": cat, "Driver": rc.label, "Direction": rc.direction, "Relative Weight": rc.weight}
                    for cat, reasons in [
                        ("Loss", explanation.loss_reasons),
                        ("Bind", explanation.bind_reasons),
                        ("Appetite", explanation.appetite_reasons),
                    ]
                    for rc in reasons
                ]
                if reason_rows:
                    st.dataframe(pd.DataFrame(reason_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No reason codes to display.")

                st.caption("Reconciliation: model base rate plus every driver's contribution "
                           "(named and pooled) reproduces the final figure exactly.")
                recon_rows = []
                for name, wf in [("Expected Loss Cost", explanation.loss_waterfall),
                                  ("Bind Probability", explanation.bind_waterfall)]:
                    if wf is None:
                        continue
                    recon_rows.append({
                        "Metric": name, "Model Base": f"{wf.base_value:,.2f}",
                        "Guardrail Adjustment": f"{wf.guardrail_adjustment:,.2f}" if abs(wf.guardrail_adjustment) > 0.01 else "-",
                        "Final (reconciled)": f"{wf.final_value:,.2f}", "Units": wf.units,
                    })
                if recon_rows:
                    st.dataframe(pd.DataFrame(recon_rows), use_container_width=True, hide_index=True)
