# Green Grid Allocator

An interactive smart-grid dispatch simulator driven by **traditional machine
learning**. Every simulated hour it forecasts city electricity demand and
renewable availability, then dispatches the generation fleet in merit order —
renewables first, then battery storage, then fossil plants by cost — to serve
demand with the least CO₂ possible.

Two interfaces are included: a web dashboard (`app.py`) and a terminal
simulator (`grid_sim.py`).

---

## Quick start (Windows)

Double-click **`run.bat`**. It creates a virtual environment, installs the
(tiny) runtime dependencies, starts the server and opens
http://127.0.0.1:8000. Linux/macOS: `./run.sh`.

Models are already trained and exported to `runtime/*.json`, so the first launch
takes seconds. If `runtime/` is ever missing, `run.bat` trains and exports
automatically. Double-click **`train.bat`** to retrain + re-benchmark on demand.

## Deploy to Vercel

1. Push this folder to a Git repo (or run `vercel` inside it).
2. Import it on vercel.com. No build command, no env vars, no settings to change.

How it works: the web app is stateless. `public/` is served as static files and
`api/index.py` is the single Python serverless function. The grid state lives in
the browser and is sent with each request (Save/Load use browser storage). The
function needs only `fastapi` because the solar/wind models are exported to JSON
(`export_runtime.py`) and evaluated in pure Python. After retraining, commit the
new `runtime/*.json` and redeploy. `.vercelignore` keeps the big CSVs and
pickles out of the deployment.

## Commands

| Command | What it does |
|---|---|
| `python train.py` | Trains all candidates, selects the best on validation, saves artifacts to `models/` |
| `python train.py --full` | Same, with heavier hyper-parameters (slower, usually better) |
| `python train.py --tolerance 0.05` | Change the accuracy band used during selection |
| `python test.py` | Full benchmark (quick mode by default) |
| `python test.py --quick` | Lighter hyper-parameters — finishes in ~1–2 minutes |
| `python test.py --full` | Heavier hyper-parameters |
| `python test.py --tolerance 0.10` | Set the accuracy-within-tolerance band |
| `python app.py` | Web dashboard on `http://127.0.0.1:8000` (what `run.bat` runs) |
| `python export_runtime.py` | Re-export trained models to `runtime/*.json` (needed after `train.py`) |
| `python grid_sim.py` | Terminal simulator |
| `python docs/generate_report.py` | Regenerates the PDF + markdown report from the saved metrics |

---

## The machine learning

### What is predicted

| Target | Type | Unit | Notes |
|---|---|---|---|
| `Zone 1 Power Consumption` | Regression | kW | The real forecasting problem |
| Solar capacity factor | Regression | 0–1 | **Synthetic** target (see Limitations) |
| Wind capacity factor | Regression | 0–1 | **Synthetic** target (see Limitations) |

### Approach

Traditional ML only — scikit-learn plus XGBoost. There is **no deep learning**
in this project: no TensorFlow, no PyTorch, and no CNN, RNN, LSTM, GRU or
transformer models.

**SIFT is not used, and would not be meaningful here.** SIFT is a computer-vision
keypoint descriptor that assumes pixels and spatial neighbourhoods. This
project's inputs are numeric readings from meters and weather sensors arranged
as rows in a table — there is no image for it to operate on. The same applies to
BERT, which operates on tokenised natural language.

The feature extraction actually used is:

- **Weather** — Temperature, Humidity, Wind Speed, general diffuse flows, diffuse flows
- **Cyclical time** — hour, day-of-week, month, each with sin/cos encodings so
  hour 23 sits next to hour 0 rather than maximally far from it
- **Lag features** — lag 24 h and lag 168 h for each of the three power zones
- **Rolling statistics** — rolling means over 24 h and 168 h windows
- **Weather lags** — lag 24/168 and a 24 h rolling mean per weather channel

41 features in total.

### Leakage prevention

- Chronological 70/15/15 split — the data are **never shuffled**
- Lags use `shift(24)` / `shift(168)`; rolling means are `.shift(1)`-ed so the
  window closes at *t*−1 and never includes the hour being predicted
- Raw Zone 1/2/3 readings for hour *t* are excluded from the feature list —
  only their history is used
- Scalers and class thresholds are fitted on training rows only
- Current-hour weather **is** used, which is standard: at forecast time it is a
  known exogenous input from a weather forecast

### Shared pipeline

`ml_pipeline.py` is the single source of truth for loading, feature
engineering, splitting, metrics and the model registry. `train.py`, `test.py`
and `grid_sim.py` all import it, so training-time and inference-time feature
definitions cannot drift apart.

---

## Metrics

**Native regression metrics:** MAE, RMSE, R², and *accuracy-within-tolerance*
— the percentage of hours forecast within ±10% of actual demand (configurable
with `--tolerance`).

**Derived demand-class metrics:** precision, recall, F1 and classification
accuracy do not exist natively for a regression task. They are derived through
a documented auxiliary view: actual and predicted demand are both binned into
Low / Medium / High using tertile thresholds computed on the **training set
only**, then macro-averaged classification metrics are computed on those bins.
They are always labelled *derived* and are never presented as regression scores.

**Gaussian Naive Bayes** appears only as a genuine classification baseline on
those Low/Medium/High labels. MAE, RMSE and R² are left blank for it because
those quantities are undefined for a classifier.

### How the best model is chosen

Ranked by a single declared rule:

1. Highest accuracy-within-tolerance
2. Then lowest RMSE
3. Then highest R²

`test.py` applies this on the **test** partition to report the best individual
model and the best ensemble. `train.py` applies the same rule on the
**validation** partition to pick the production model — selecting on test
results would leak the test set. The production model may therefore differ from
the test-set winner; that is the expected, honest consequence of keeping the
test period untouched.

### Ensembles

Three weighted ensembles are built (Random Forest + XGBoost, Extra Trees +
XGBoost, and a top-3 blend). Weights are non-negative, sum to 1, and are fitted
on the validation partition by minimising validation RMSE over the simplex, then
applied unchanged to the test predictions. When members are highly correlated the
optimiser can legitimately put all weight on one member — the benchmark reports
that plainly instead of forcing an artificial blend.

---

## Web dashboard

All the original simulator behaviour is unchanged: live polling, KPI cards,
charts, the source table, the control dock, weather and demand controls, and
save / load / import / export.

One panel is added — **Model Performance** — which reads
`models/benchmark_results.json` via `GET /api/model-performance` and displays
the real benchmark: the top models plus the best ensemble, the selected best
individual and best combination with a data-driven explanation, and the derived
classification metrics clearly marked. Nothing in it is hard-coded. If the
benchmark has not been run, the panel shows a "run `python test.py`" hint and
the rest of the dashboard is unaffected.

### API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/state` | GET / POST | Default snapshot / recompute for the posted `state` |
| `/api/health` | GET | Heartbeat for the LIVE badge |
| `/api/time` | POST | Advance/rewind hours, or jump to a datetime |
| `/api/demand` | POST | Change demand by `delta` or set absolutely |
| `/api/weather` | POST | Override weather fields |
| `/api/sources` | POST | Add a generation source |
| `/api/sources/remove` | POST | Remove a source by id |
| `/api/import` | POST | Load a grid JSON (save / load / export happen in the browser) |
| `/api/model-performance` | GET | **New** — benchmark results for the UI panel |

---

## Project layout

```
mlpbl/
├── ml_pipeline.py          Shared preprocessing, features, splits, metrics, registry
├── train.py                Trains and saves the production models
├── test.py                 Full benchmark + ensembles → JSON/CSV
├── grid_sim.py             Simulator core, terminal UI, production inference
├── run.bat / run.sh        One-click launcher
├── train.bat               Retrain + benchmark + export
├── app.py                  Local launcher (opens the browser)
├── api/index.py            FastAPI app (also the Vercel function)
├── api/_core.py            Stateless simulator core (pure Python)
├── export_runtime.py       Models -> runtime/*.json
├── runtime/                JSON models + weather table used by the web app
├── vercel.json, .vercelignore
├── requirements.txt        Runtime (fastapi, uvicorn)
├── requirements-train.txt  Training / benchmark stack
├── public/                 index.html, app.js, style.css
├── models/                 Artifacts, metadata, benchmark results
│   ├── demand_model.pkl, solar_model.pkl, wind_model.pkl
│   ├── scaler_*.pkl, features.json, last_weather.json
│   ├── model_metadata.json
│   └── benchmark_results.json / .csv
├── docs/
│   ├── ML_Model_Implementation_Report.pdf
│   ├── ML_Model_Implementation_Report.md
│   ├── generate_report.py  Editable source for the report
│   └── fonts/              Bundled DejaVu fonts (+ licence)
└── Tetuan City power consumption.csv
```

`Steel_industry_data.csv`, `smart_grid_stability_augmented.csv` and
`household_power_consumption.txt` are retained from the original project for
reference; the training, benchmark and inference code does not use them.

---

## Limitations

- **Solar and wind targets are synthetic.** The dataset contains no measured
  renewable generation. Solar capacity factor is derived from diffuse-flow
  readings plus a solar-altitude and seasonality model; wind capacity factor
  comes from a turbine power curve applied to wind speed. Their very high R²
  mainly shows the models learned a deterministic formula — it is **not**
  evidence of real renewable forecast skill. Only the demand metrics reflect a
  genuine prediction problem.
- **Classification metrics are derived**, measuring band placement rather than
  kilowatt accuracy, and are easier to score well on than RMSE.
- **One city, one year** (Tetouan, 2017), so a recurring seasonal pattern
  cannot be distinguished from a one-off event that year.
- **The test period is winter** — a chronological split puts November–December
  in the test set, so metrics lean toward cold-season behaviour.
- **One-step-ahead framing.** Multi-day-ahead forecasting would need recursive
  prediction and would show materially worse error.
- **Weather is assumed known** for the target hour; in live operation its own
  forecast error would compound.

Full detail is in `docs/ML_Model_Implementation_Report.pdf`.
