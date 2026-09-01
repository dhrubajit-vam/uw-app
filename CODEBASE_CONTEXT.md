# Codebase Context — US Personal Auto Underwriting & Pricing App

Reference doc for making changes to this repo without re-deriving the architecture from scratch.
Written after a full read of every source file. Paths below are repo-relative.

---

## 1. What this app is

A Streamlit app that replaces an Excel prototype's (`US_Personal_Auto_UW_Pricing_Prototype_dh.xlsx`)
hand-coded underwriting formulas with real trained ML models, wired into the same two-tab
layout (**Dashboard**, **Interface**) and the same deterministic premium/eligibility rules.

Three independently trained ML models feed a non-ML, fully auditable rules engine. Nothing
downstream of the models is "learned" — hard-stops, premium loads, and score weights are
hard-coded business policy. An optional Azure OpenAI layer narrates the finished numbers in
prose; it never computes or alters anything.

```
Submission (form input or historical row)
   -> 3 ML models (frequency, severity, bind probability)   [scoring/predictors.py]
   -> deterministic rules engine (hard-stops, appetite, premium, composite score, band)
                                                              [rules/rules_engine.py]
   -> explainability (exact Shapley reason codes + waterfalls)
                                                              [scoring/explain.py]
   -> advisor (coverage-fit comparison, decline alternatives, re-simulated not fabricated)
                                                              [scoring/advisor.py]
   -> AI narrative (optional, narrates only, falls back to None on any failure)
                                                              [scoring/ai_narrator.py]
   -> Streamlit UI (Dashboard tab + Interface tab)
                                                              [app/app.py]
```

**Governance principle baked into the code, not just docs**: loss models never see
bind/behavior features, the bind model never sees loss/claims features, and no model ever
sees another model's output fed back in. Enforced in `train/feature_config.py`'s three
disjoint feature lists (`LOSS_*`, `BIND_*`, `OUTCOME_FIELDS`). Preserve this separation in
any change — don't let a feature leak across model boundaries.

---

## 2. Directory map

```
data/
  export_from_xlsx.py          xlsx -> scored_dataset_full.parquet (run once, only script touching .xlsx)
  scored_dataset_full.parquet  22,000-row training set (generated)
  model_scored_dataset.parquet full book scored by trained models (generated, Dashboard reads this)
train/
  feature_config.py            LOSS_FEATURES / BIND_FEATURES / OUTCOME_FIELDS — single source of truth
  preprocessing.py             build_preprocessor() (impute+onehot), engineer_bind_features()
  splits.py                    time_based_split() — 80/20 by Quote_Date, not random
  train_frequency_model.py     XGBoost Poisson regressor -> frequency_model.pkl
  train_severity_model.py      statsmodels Gamma GLM (log link) -> severity_model.pkl
  train_bind_model.py          LightGBM classifier -> bind_model.pkl
  evaluate_models.py           console-only lift/calibration report (not shown in app)
models/                        trained model + preprocessor .pkl artifacts, plus:
  market_adj_by_state.json     avg historical Market_Adjustment per state (business lookup)
  theft_risk_by_state.json     baseline vehicle theft-risk score per state
rules/
  rules_engine.py              hard-stops, appetite score, premium build, composite score, band/action
scoring/
  predictors.py                loads the 3 models, exposes predict_frequency/severity/bind_probability
  row_mapper.py                SubmissionInputs <-> raw model-feature-row, BOTH directions
  advisor.py                   score_live_submission(), build_alternatives(), coverage_fit_analysis()
  explain.py                   exact SHAP reason codes + waterfalls (loss, bind, appetite)
  ai_narrator.py                optional Azure OpenAI narrative layer, explains only, graceful fallback
  score_book.py                batch-scores all 22k rows -> model_scored_dataset.parquet (Dashboard data)
  scenarios.py                 5 pre-built demo scenarios (real rows verified to land on stated outcome)
app/
  app.py                       Streamlit entrypoint: Dashboard tab + Interface tab, all UI/CSS
docs/
  technical_overview.md        architecture rationale (why 3 models, why not NN, etc.) — read this too
  demo_guide.html               presenter-facing walkthrough
.env.example                    Azure OpenAI config template (.env is git-ignored)
requirements.txt
```

---

## 3. The three ML models

| Model | Algorithm | Target | Trained on | File |
|---|---|---|---|---|
| Frequency | `xgboost.XGBRegressor`, `objective="count:poisson"` | `Claim_Count` | `Bound_Flag==1` rows only | `train/train_frequency_model.py` |
| Severity | `statsmodels` GLM, Gamma family, log link | `Incurred_Loss / Claim_Count` | `Bound_Flag==1 & Claim_Count>0` | `train/train_severity_model.py` |
| Bind | `lightgbm.LGBMClassifier` | `Bound_Flag` | full book | `train/train_bind_model.py` |

All three: median-impute numerics / most-frequent-impute + one-hot categoricals
(`train/preprocessing.py::build_preprocessor`, `handle_unknown="ignore"`), fitted preprocessor
pickled alongside the model so live inference uses byte-identical transforms. Time-based
80/20 split by `Quote_Date` (`train/splits.py`), not random.

**Inference guardrail clips** (`scoring/predictors.py`):
- frequency clipped to `[0.005, 1.5]`
- severity clipped to `[$500, $100,000]`
- These clips are surfaced explicitly as a step in the explainability waterfall
  (`Waterfall.guardrail_adjustment` in `scoring/explain.py`) rather than hidden — don't silently
  drop this if touching predictors.py.

**Bind model needs `Final_Quoted_Premium`**, which isn't known until after the rules engine
runs — hence the two-pass scoring in `scoring/advisor.py::score_live_submission`: pass 1 with a
placeholder `bind_probability=0.5` computes the premium, pass 2 re-scores with the real bind
probability computed from that premium. Any new code path that scores a submission end-to-end
must follow this same two-pass pattern, not call `rules_engine.score_submission` directly.

Retraining: rerun `export_from_xlsx.py` (if source data changed) → all 3 `train_*.py` →
`evaluate_models.py` (sanity check lift/calibration) → `score_book.py` (regenerates the
Dashboard's parquet). `st.cache_resource`/`st.cache_data` cache for the life of the Streamlit
process — restart the app after retraining.

---

## 4. Feature lists (`train/feature_config.py`)

- `LOSS_NUMERIC_FEATURES` / `LOSS_CATEGORICAL_FEATURES` → `LOSS_FEATURES`: used by both
  frequency and severity models. Adding a new loss-relevant field means adding it here AND to
  `row_mapper.submission_inputs_to_model_row` AND (if user-entered) to a form field in `app.py`.
- `BIND_NUMERIC_FEATURES` / `BIND_CATEGORICAL_FEATURES` → `BIND_FEATURES`: includes the
  engineered `Premium_Change_Percentage_Input` (computed in
  `preprocessing.engineer_bind_features`, NOT a raw column — must be called before any bind
  prediction or bind-feature transform).
- `OUTCOME_FIELDS`: `Bound_Flag, Claim_Flag, Claim_Count, Incurred_Loss, Earned_Premium,
  Realized_Loss_Ratio` — never valid as a model input, only as a training target.

If a change adds a new raw column, decide up front which bucket it belongs to (loss-only,
bind-only, or neither) and never let it cross — that's the leakage boundary the whole app is
built around.

---

## 5. `rules/rules_engine.py` — the deterministic core

Pure business logic, intentionally non-ML. Constants live at the top of the file — **change
values there, never scatter magic numbers elsewhere**:

```python
TREND = 1.06; DEVELOPMENT = 1.03
LAE_LOAD = 0.10; EXPENSE_LOAD = 0.18
TARGET_PROFIT_MARGIN = 0.05        # UI-adjustable 2-7% via a slider
BASE_MARKET_MARGIN = 0.09
PREMIUM_FLOOR = 350; PREMIUM_CEIL = 16000
RENEWAL_BAND_LOW = 0.60; RENEWAL_BAND_HIGH = 1.90   # renewal premium held to 60%-190% of last year
TECHNICAL_FLOOR_PCT = 0.90         # discretionary adjustments can't price >10% under technical
COMPOSITE_THRESHOLDS = {"preferred": 72, "standard": 55, "non_standard": 38}
COMPOSITE_WEIGHTS = {"loss": 0.50, "bind": 0.20, "appetite": 0.30}
NON_NEGOTIABLE_HARD_STOPS = {"DUI/DWI conviction", "License suspension", "Confirmed fraud indicator"}
COVERAGE_APPETITE_BONUS = {...}    # 5 coverage options, UM/UIM Only=18 down to Full Coverage=0
```

### Pipeline inside `score_submission()`

1. `check_hard_stops(x)` → `(bool, list[str])`. Triggers: DUI, license suspension, coverage
   lapse > 60 days, confirmed fraud, undeclared business use, vehicle value > $100k, state
   outside appetite footprint, prior total loss. Non-negotiable subset (no underwriter
   discretion) is `NON_NEGOTIABLE_HARD_STOPS`; every other hard-stop reason IS negotiable and
   gets an alternative in `scoring/advisor.py`.
2. `compute_appetite_score(x, hard_stop)` → 0 if hard-stopped, else `100 - penalty +
   coverage_bonus` clipped to `[0,100]`. Penalty terms (see also `scoring/explain.py::
   explain_appetite` which must mirror this EXACTLY — it's a decomposition, not an
   approximation):
   - `4 × prior_claims_3y`, `5 × at_fault_claims_3y`, `3 × moving_violations_3y`
   - `15` if major violation
   - `10` if coverage lapse strictly between 30 and 60 days
   - `max(0, (vehicle_value − 80,000) / 2500)`
   - `max((cat_exposure_score − 7) × 4, 0)`
   - `5` if driver under 25
   - `5` if declared business use (not undeclared)
   - `max((litigation_environment − 7) × 3, 0)`
   - `6` if 3+ prior claims in 5 years
3. `expected_loss_cost = frequency × severity`; `expected_loss_load = loss_cost × TREND ×
   DEVELOPMENT`.
4. `compute_cat_load` (0.02–0.08 range off CAT exposure) and `compute_risk_load` (clipped
   0.02–0.06, adds for renewal/performance-vehicle/high-value/young-driver).
5. `compute_technical_premium`: `loss_load × (1+LAE+EXPENSE+cat+risk) / (1 − target_margin)`.
6. `compute_business_premium`: applies market adj (state lookup), competitive adj (clipped
   ±5–6% vs a competitor estimate if provided), retention adj (renewals only, tenure +
   multi-policy), channel adj (Aggregator −2%, Agent +1%), surcharge adj (major violation,
   performance vehicle, 30–60 day lapse), then subtracts bundle/safe-driver/telematics
   discounts. Floored at `technical_premium × TECHNICAL_FLOOR_PCT`.
7. `apply_premium_guardrails`: for renewals with a `prior_year_premium`, clamps to the
   **median of** `[raw, prior×0.60, prior×1.90]` (i.e. a soft band, not a hard clip — read
   `sorted(...)[1]` carefully if touching this), then clips to `[PREMIUM_FLOOR, PREMIUM_CEIL]`.
   A `non_standard_load` multiplier (e.g. 1.15 for a 15% price-up) is applied AFTER guardrails.
8. `compute_composite_score`: `0.50×(100−loss_propensity) + 0.20×bind_propensity_score +
   0.30×appetite_score`, rounded to 1 decimal. Note loss propensity is *inverted* (lower loss
   risk → higher composite contribution).
9. `band_for_score`: Hard Stop (if hard_stop) else Preferred (≥72) / Standard (≥55) /
   Non-Standard (≥38) / Decline.
10. `recommend_action`: maps band → one of `Auto-Quote / STP`, `Light UW Review`,
    `Thorough Verification`, `Refer to Underwriter (Rate-Adequacy Exception)` (Preferred but
    final premium < technical), `Decline` (loss ratio > 1.05), `Refer to Senior Underwriter`,
    or the hard-stop route string.

`ScoringResult` is a flat dataclass — the app's "Technical & audit detail" tab dumps
`result.__dict__` directly, so any new field added there automatically shows up in the audit
table (stringified) with no extra UI work.

`SubmissionInputs` is the canonical form-input schema (matches the Interface tab 1:1). Fields
not collected by the UI (`agent_engagement`, `digital_engagement`) default to `None` and are
imputed downstream — see §7.

---

## 6. `scoring/row_mapper.py` — the schema bridge

Single source of truth mapping `SubmissionInputs` (business objects) ↔ raw model-feature rows
(column names matching the training parquet). Two directions:

- `submission_inputs_to_model_row(x) -> dict`: used by live scoring (app.py, advisor.py,
  explain.py). Starts from `_MODEL_ROW_DEFAULTS` (a dict of NaN/None for every feature the
  Interface form doesn't collect), then overwrites with the real submission values.
- `row_to_submission_inputs(row, ...) -> SubmissionInputs`: used by `score_book.py` to convert
  historical parquet rows back into the business object for rules-engine scoring.

**Critical gotcha, already learned the hard way (see comment in the file)**: fields not
collected by the live form (weather/hail/flood/hurricane risk, telematics behavior metrics,
months since last claim, etc.) MUST be passed as `float("nan")`/`None`, never a hand-picked
"midpoint" default. The preprocessor's `SimpleImputer` already resolves NaN to the
population-median/most-frequent value from training — a hand-picked default (e.g. risk=5 when
the true median is ~2–3, or Months_Since_Last_Claim=999 vs a real median of 61) measurably
distorted every live prediction toward high risk. **If you add a new UI-uncollected feature,
default it to NaN/None in `_MODEL_ROW_DEFAULTS`, not a guessed value.**

Coverage selection maps through `COVERAGE_FLAGS` (in `rules_engine.py`) to the actual
`Collision_Flag`/`Comprehensive_Flag`/`UM_UIM_Flag` the loss models read — switching coverage
genuinely changes the predicted loss cost, not just an appetite bonus.

---

## 7. `scoring/predictors.py` — model loading

`ModelBundle` loads all 6 artifacts (3 models + 3 preprocessors) once. `get_model_bundle()` is
a module-level singleton, wrapped by `st.cache_resource` in `app.py::load_models()`. Three
predict methods, each applying the correct feature subset and (for severity) `sm.add_constant`
for the GLM's intercept, and (for bind) `engineer_bind_features` first.

---

## 8. `scoring/advisor.py` — live scoring + alternatives

- `score_live_submission(bundle, x, book, market_adj_lookup, non_standard_load=0.0)`: the
  two-pass scorer described in §3. `book["Pred_Expected_Loss_Cost"]` (from the pre-scored
  book) is used to compute `loss_propensity_score` as a **percentile rank** of this
  submission's loss cost against the whole historical book — so this function needs the
  scored book loaded, not just the models.
- `coverage_fit_analysis(...)`: re-scores the SAME risk under all 5 `COVERAGE_OPTIONS` via
  `dataclasses.replace(x, coverage_type=option)`, ranks by `(band_rank, composite_score)`,
  and only recommends a switch if it changes band or gains ≥`MIN_COMPOSITE_GAIN` (2.0) points.
  Hard-stopped submissions always report "coverage can't fix this."
- `build_alternatives(bundle, x, base_result, book, market_adj_lookup)`: only fires for
  `Decline` or `Hard Stop` bands, returns up to 3 `Alternative` objects sorted by composite
  score improvement (re-simulated alternatives first, referral-only notes last). Hard-stop
  alternatives (`_hard_stop_alternatives`) only exist for negotiable reasons (undeclared
  business use → reclassify; lapse > 60 days → conditional monitored acceptance; vehicle value
  over limit → specialty placement referral; state outside footprint → E&S referral; prior
  total loss → manual override after inspection). Decline alternatives
  (`_decline_alternatives`): require telematics (if claims/violations/young + not enrolled),
  raise deductibles (if high-value/CAT + low deductibles), or always-available price-up at
  `NON_STANDARD_LOAD = 0.15`. **Every alternative that has a `result` field is a genuine
  re-score through `score_live_submission` — never fabricate a number here.**

---

## 9. `scoring/explain.py` — exact explainability

Not approximated:
- Loss reasons: XGBoost's native `pred_contribs=True` (frequency) + Gamma GLM's raw
  `coefficient × value` (severity), both in log-link space, summed feature-by-feature since
  `log(freq) + log(sev) = log(expected loss cost)` exactly.
- Bind reasons: LightGBM's native `pred_contrib=True`, logit space, sigmoid-mapped back to %.
- Appetite reasons: literal re-statement of `compute_appetite_score`'s penalty terms — must
  stay in exact sync with `rules_engine.compute_appetite_score` if that function changes.

Key internals:
- `_humanize(raw_name, categorical_cols)`: turns `"num__Driver_Age"` → `"Driver Age"` and
  `"cat__Vehicle_Use_Business"` → `"Vehicle Use: Business"`, with special-casing for `_Flag`
  columns to read as Yes/No rather than 0/1.
- `_raw_contribs(...)`: features that were NaN for this submission (never observed, e.g.
  telematics behavior for an unenrolled driver) are excluded from named reasons via
  `skip_names` but their contribution is still pooled into `skipped_total` so the waterfall's
  additivity guarantee holds — **don't just drop unknown-feature contributions, they must be
  accounted for somewhere or the numbers stop reconciling.**
- `_build_waterfall(...)`: `base + sum(all contributions) = final`, in link space, then mapped
  through `link()` to human units. `guardrail_adjustment` captures any gap between the raw
  model output and the predictors.py clip — surfaced, not hidden.
- `confidence_level(result)`: distance from the nearest composite-score band threshold — High
  (≥10 pts away), Medium (≥4), Low (<4). Hard stops are always "High" confidence.
- `_combine_top_reasons`: priority order appetite → loss → bind (mirrors stated governance:
  "hard-stops override all statistical scores; loss dominates; bind never drives approval
  alone").

`explain_submission(bundle, x, result)` is the single entrypoint called from `app.py`.

---

## 10. `scoring/ai_narrator.py` — optional narrative layer

Azure OpenAI (`gpt-4o` by default) via the `openai` SDK's `AzureOpenAI` client. Config read
from env vars (`.env`, git-ignored) first, then `st.secrets` (for Streamlit Cloud deploys).
**Contract: every public function returns `None` on ANY failure** (missing config, network
error, bad deployment name, timeout) — callers (`app.py`) must always handle `None` by falling
back to existing rule-based text, never assume the AI call succeeds. `is_available()` gates
whether the app even attempts a call (avoids a wasted spinner when unconfigured).

`_GROUNDING_RULES` is the system prompt: the model is instructed to *only* phrase/prioritize
facts it's given, never invent numbers, no em-dashes (stripped again client-side via
`_strip_dashes` as a belt-and-braces), no AI self-disclosure, no bullet points, 3–5 sentences.
`narrate_decision()` and `narrate_alternatives()` build a fact list from `ScoringResult` /
`Explanation` / `Alternative` objects and hand it to `_complete()`. **If you add a new fact
that should appear in the narrative, add it to the `facts` list in these functions — the
prompt template itself shouldn't need to change.**

---

## 11. `scoring/score_book.py` — batch scoring for the Dashboard

Reads `data/scored_dataset_full.parquet`, runs all 3 models over every row (bind probability
uses each row's own historically-recorded `Final_Quoted_Premium`/`Prior_Year_Premium` as the
feature input, since this is scoring what actually happened, not a live what-if — so no
two-pass placeholder is needed here the way the live `advisor.score_live_submission` path
requires), computes `Pred_Expected_Loss_Cost` and a percentile-rank `Loss_Propensity_Score`,
builds/refreshes `models/market_adj_by_state.json` from the historical `Market_Adjustment`
column average per state, then row-by-row converts each row to `SubmissionInputs`
(`row_mapper.row_to_submission_inputs`) and calls `rules_engine.score_submission` directly
(single-pass). Output is `data/model_scored_dataset.parquet`, the concatenation of original
columns plus every `ScoringResult` field. This is what `app.py::load_scored_book()` reads for
the Dashboard tab.

Run after training, before launching the app, and after any retrain.

---

## 12. `scoring/scenarios.py` — demo scenarios

5 real rows drawn from the book, each verified to land on its stated outcome via the live
scoring path (not pre-computed/hard-coded results — the app re-scores them like any hand-typed
submission). `BASE` is also what "Reset" restores. If you need a 6th scenario, pull an actual
row from the book and re-verify against the live app rather than hand-inventing values — the
file's own docstring warns that hand-invented combinations land outside the training
distribution and can misbehave.

---

## 13. `app/app.py` — the Streamlit UI

~1300 lines, no logic beyond orchestration + presentation — all real computation happens in
`rules/`, `scoring/`. Structure:

1. **Imports/paths**: inserts `scoring/`, `rules/`, `train/` onto `sys.path` (no package
   installs — this repo is run as flat scripts, not an installed package).
2. **Theme/CSS** (lines ~63–466): color constants (`NAVY`, `ACCENT`, `SHAP_RED`/`SHAP_BLUE` for
   the SHAP-standard red/blue driver charts vs. the app's own red/green good/bad palette
   elsewhere), a big injected `<style>` block, a custom Plotly template.
3. **Helper components**: `kpi_card`, `section_title`, `ai_card`, `recommendation_card`,
   `driver_bar_chart` (loss/bind — direction-colored, SHAP red/blue), `appetite_bar_chart`
   (penalties only, always red).
4. **Cached loaders**: `load_models()` (`st.cache_resource`), `load_scored_book()`,
   `load_market_adj()`, `load_theft_risk_by_state()` (all `st.cache_data`).
5. **`compute_vehicle_theft_risk(state, anti_theft_available, lookup)`**: theft-risk is
   derived (state baseline − 1.5 if anti-theft device present), not a raw user slider anymore.
6. **Dashboard tab**: 5 filters (state/band/channel/new-renewal/body-type) → 6 KPI cards → 4
   charts (band distribution, action mix, avg composite by state, technical-vs-final premium
   scatter) → rate-adequacy exceptions table.
7. **Interface tab**: this is the complex part.
   - `_iv(key, fallback)`: reads a form input's current value from `session_state`, respecting
     a loaded scenario.
   - `_cb(container, label, field, default, **kwargs)`: checkbox helper — **must** go through
     `session_state` with an explicit key because `st.checkbox` doesn't reliably re-apply a
     changed `value=` after the user has ever toggled it manually (documented gotcha in the
     code, not a Streamlit bug you should try to "fix" — work around it the same way for any
     new checkbox).
   - `_load_scenario()` / `_reset_scenario()`: wired to the scenario selectbox's `on_change`
     and the Reset button's `on_click`.
   - **Section split rationale**: sections 1–3 (Driver & Loss Risk, Vehicle & Coverage,
     Telematics) live OUTSIDE `st.form(...)` because the Telematics safety-score slider needs
     to appear/disappear live as the "enrolled" checkbox is toggled — `st.form` batches every
     widget until submit, so no widget inside a form can react to another widget's change
     before that. Sections 4–6 (Claims History, Appetite & Eligibility, Bind & Market) ARE
     inside the form since none of them need that live behavior, and batching avoids
     re-scoring on every slider drag. **If a new input needs live cross-widget reactivity, it
     must go outside the form; otherwise put it inside to avoid gratuitous reruns.**
   - `MAX_TRAINED_DRIVER_AGE = 85`: ages above this still work (the app doesn't block them)
     but show an amber warning that the model is extrapolating beyond training data.
   - On submit, builds a `SubmissionInputs`, calls `advisor.score_live_submission`, then
     `explain.explain_submission`. Result stored via `st.session_state["has_result"]` so the
     result section persists across reruns (e.g. the live profit-margin slider) without
     needing to re-click Calculate.
   - **Profit margin slider is live**: keyed directly to `st.session_state["live_margin_pct"]`,
     read via `setdefault` before the widget renders (same session-state-first pattern as
     `_iv`), so dragging it re-scores immediately without a form resubmit.
   - Coverage Fit table, AI narrative cards (conditionally rendered if
     `ai_narrator.is_available()`), "Key Factors Behind This Decision" expander (chips +
     3 driver bar charts), Underwriter Recommendations (only for Decline/Hard Stop bands),
     "Technical & audit detail" expander (full `ScoringResult.__dict__` dump + reason-code
     table + reconciliation table).

**Color convention held consistently**: blue = helps the outcome, red = hurts it, across all
three driver charts (appetite chart is red-only since penalties are never positive).

---

## 14. Running the app

```bash
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# One-time / after source data changes:
python data/export_from_xlsx.py --source /path/to/workbook.xlsx

# After any model/feature change:
python train/train_frequency_model.py
python train/train_severity_model.py
python train/train_bind_model.py
python train/evaluate_models.py     # console sanity check — don't skip
python scoring/score_book.py        # regenerates Dashboard data

streamlit run app/app.py            # http://localhost:8501
```

No `.git` repo currently exists in this working directory (per environment info) — if
initializing one is needed for version control, confirm with the user first.

---

## 15. Conventions to preserve when making changes

- **No inline "fix"/"changed for X" comments** — this codebase's comments explain *why*
  (a non-obvious constraint, a past bug, a workaround), never *what* or *when*. Match that
  style; don't narrate the task in comments.
- **Auditability over cleverness**: every number the UI shows must be traceable to a rule or
  an exact Shapley/GLM contribution. Never introduce an approximation, heuristic guess, or
  LLM-generated number into anything but the narrative text.
- **Single source of truth, not duplicated logic**: feature lists live only in
  `feature_config.py`; the input↔row mapping lives only in `row_mapper.py`; rules constants
  live only at the top of `rules_engine.py`. If a change needs the same fact in two places,
  something is structured wrong — import it instead.
- **NaN/None for genuinely unknown fields, never a guessed default** — see §7's documented
  incident. This applies to any new field the live form doesn't collect.
- **`dataclasses.replace(x, field=value)`** is the established pattern for "same submission,
  one field changed" (used throughout `advisor.py` for coverage fit and alternatives) — reuse
  it rather than manually reconstructing a `SubmissionInputs`.
- **Two-pass scoring for any new live-scoring path** that needs bind probability (see §3/§5) —
  don't call `rules_engine.score_submission` once and assume bind_probability is correct if
  `Final_Quoted_Premium` wasn't known yet.
- **AI narrator failures are silent by design** — never let an AI-layer change make a `None`
  return propagate into an exception; every caller must keep working with zero AI narrative.
- Python: `from __future__ import annotations` used throughout for forward-ref dataclass
  typing; keep that pattern in new modules under `rules/`/`scoring/`.

---

## 16. Likely extension points (for reference, not a task list)

- **New rating factor**: add to `feature_config.LOSS_FEATURES` (or `BIND_FEATURES`), add a
  default in `row_mapper._MODEL_ROW_DEFAULTS` (NaN if not user-collectible), add a mapping
  line in `submission_inputs_to_model_row` and `row_to_submission_inputs`, add a form field in
  `app.py` if user-facing, retrain all affected models.
- **New hard-stop rule**: add a condition + reason string in `check_hard_stops`; decide if it
  belongs in `NON_NEGOTIABLE_HARD_STOPS`; if negotiable, add a branch to
  `advisor._hard_stop_alternatives`.
- **New underwriter alternative**: add to `_decline_alternatives` or
  `_hard_stop_alternatives` in `advisor.py`, following the existing pattern of
  `dataclasses.replace` + `score_live_submission` for anything with a concrete re-rating, or a
  referral-only `Alternative` (no `result`) for a process-only option.
- **New coverage option**: extend `COVERAGE_APPETITE_BONUS` and `COVERAGE_FLAGS` together in
  `rules_engine.py` — they must stay in lockstep (same keys).
- **New Dashboard chart/filter**: Dashboard tab reads only from
  `data/model_scored_dataset.parquet` columns (i.e., `ScoringResult` fields + original book
  columns) — no new model call needed, just a new `px.*`/`go.*` block following the existing
  `chart-card` wrapper pattern.
