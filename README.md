# US Personal Auto: Underwriting & Pricing (Model-Backed)

Replaces the source workbook's (`US_Personal_Auto_UW_Pricing_Prototype_dh.xlsx`)
hand-coded frequency/severity/bind formulas with real trained models, wired
into the same two-tab layout (Dashboard, Interface) and the same
premium/eligibility rules engine.

| Component | Model |
|---|---|
| Claim Frequency | XGBoost, `count:poisson` objective |
| Claim Severity | Gamma GLM (statsmodels) |
| Bind Propensity | LightGBM Classifier |
| Appetite / Hard-Stops / Premium build / Composite score | Deterministic rules engine (unchanged from workbook's documented logic: kept auditable on purpose) |

## Project layout

```
uw_app/
├── data/
│   ├── export_from_xlsx.py         # xlsx -> scored_dataset_full.parquet
│   ├── scored_dataset_full.parquet # 22,000-row training set (generated)
│   └── model_scored_dataset.parquet# full book scored by trained models (generated)
├── train/
│   ├── feature_config.py           # feature lists per model (no-leakage rules)
│   ├── preprocessing.py            # shared imputers/encoders
│   ├── splits.py                   # time-based train/test split
│   ├── train_frequency_model.py
│   ├── train_severity_model.py
│   ├── train_bind_model.py
│   └── evaluate_models.py          # CONSOLE-ONLY performance report
├── models/                         # trained model + preprocessor artifacts (generated)
├── rules/
│   └── rules_engine.py             # hard-stops, appetite score, premium build, composite score
├── scoring/
│   ├── predictors.py               # loads models, exposes predict_*() methods
│   ├── row_mapper.py               # SubmissionInputs <-> raw model row, both directions
│   ├── advisor.py                  # live scoring + underwriter alternative-recommendation engine
│   ├── explain.py                  # Shapley reason codes + contribution decompositions
│   ├── ai_narrator.py              # optional Azure OpenAI narrative layer (explains, never decides)
│   └── score_book.py               # batch-scores all 22k rows for the Dashboard tab
├── app/
│   └── app.py                      # Streamlit entrypoint (Dashboard, Interface tabs)
└── requirements.txt
```

## Setup

```bash
cd uw_app
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run: step by step

**1. Export the training data from the source workbook (run once):**
```bash
python data/export_from_xlsx.py --source /path/to/US_Personal_Auto_UW_Pricing_Prototype_dh.xlsx
```

**2. Train all 3 models:**
```bash
python train/train_frequency_model.py
python train/train_severity_model.py
python train/train_bind_model.py
```

**3. Check model performance in the console** (lift charts, deviance, AUC, calibration: this is NOT shown in the app, it's a developer/actuarial sanity check):
```bash
python train/evaluate_models.py
```
Review the lift tables: `avg_actual` should rise roughly monotonically across deciles for frequency/severity, and the bind model's calibration table should show predicted ≈ actual per decile. Don't proceed to the app if these look broken.

**4. Score the full historical book** (produces the dataset the Dashboard tab reads):
```bash
python scoring/score_book.py
```

**5. Launch the app:**
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501` with two tabs:
- **Dashboard**: book-wide KPIs, filterable by state/band/channel/new-renewal/body type, charts, rate-adequacy exception list.
- **Interface**: live submission form; click **Calculate Risk Score** to score a hypothetical policy through the 3 trained models + rules engine. Returns the full underwriting decision plus:
  - **Suggestions**: for a Decline or Hard-Stop outcome, `scoring/advisor.py` re-simulates concrete alternatives (modify terms, price up to non-standard, or conditional acceptance) through the same rules engine rather than just returning a decline. A hard-stop only has *no* alternative when every triggered reason is regulatory/fraud in nature (DUI, license suspension, confirmed fraud).
  - **Explainability**: `scoring/explain.py` surfaces the top reason codes behind the loss, bind, and appetite scores. The loss/bind reason codes are exact per-instance Shapley decompositions (XGBoost's and LightGBM's own native `pred_contribs`, not an approximation), the appetite reason codes are an exact decomposition of the deterministic penalty formula, and a plain-English summary + decision-confidence rating tie it together.
  - **AI narrative** (optional): if Azure OpenAI is configured, `scoring/ai_narrator.py` turns the finished figures into a readable paragraph for both the decision and the recommendations. See "AI narrative layer" below.

## Explainability: how the charts work

Each panel (Loss Risk / Bind Propensity / Appetite & Eligibility) is one
plain vertical bar chart: bars above the line make the outcome worse, bars
below make it better, ranked by size. No relativities, no log space, no
waterfall math on screen, the goal is "what drove this" readable at a
glance. (An earlier version showed a log-scale relativity chart and a
waterfall side by side; it was mathematically defensible but unreadable in
a live demo, so it was replaced with this.)

The underlying computation is still exact Shapley decomposition (XGBoost's
and LightGBM's own native `pred_contribs`, the same mechanism the
third-party `shap` package wraps for tree models) and it's still available
in full, additive, dollar-reconciled form in the "Technical & audit detail"
expander for anyone who wants the underlying figures.

Two safeguards worth knowing:
- Features that were never observed for a submission (e.g. telematics
  metrics for a driver who isn't enrolled) are excluded from the named
  reason codes but still pooled into "all other factors", so the
  decomposition keeps adding up to the real prediction.
- The guardrail clips in `scoring/predictors.py` (frequency to
  `[0.005, 1.5]`, severity to `[$500, $100k]`) are surfaced as their own
  step rather than hidden, so every chart reconciles exactly to the number
  the rules engine used.

## AI narrative layer (optional)

Copy `.env.example` to `.env` and fill in the Azure OpenAI values. `.env` is
git-ignored; never commit real credentials.

The AI **explains, it never decides**. Every band, premium, score, and
reason code is computed by the models and rules engine *before* the AI is
called; `ai_narrator.py` passes those finished figures in as ground truth
and instructs the model to phrase and prioritize them without introducing
anything new. If the service is unreachable, misconfigured, or errors for
any reason, every call returns `None` and the app falls back to the
rule-based text with nothing missing. A demo never breaks on a failed AI
call.

## Retraining

Whenever the underlying data changes, rerun steps 1–4 in order: the app
only ever reads the generated parquet/pkl artifacts, so it always reflects
the most recently trained models after a restart (`st.cache_resource` /
`st.cache_data` cache for the life of the Streamlit process: restart the
app after retraining to pick up new artifacts).

## Notes on the rules engine

`rules/rules_engine.py` intentionally stays non-ML: hard-stop eligibility
rules, premium loads (LAE%, Expense%, profit margin), and composite-score
weights (50% loss / 20% bind / 30% appetite) are configuration constants at
the top of the file. Change them there, not in the models.
