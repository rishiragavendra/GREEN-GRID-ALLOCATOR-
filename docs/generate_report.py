"""Generate docs/ML_Model_Implementation_Report.pdf.

Every number in the report is read from the artifacts produced by the real
runs (`models/model_metadata.json` and `models/benchmark_results.json`).
Nothing is hard-coded, so regenerating after a new run keeps the document
truthful:

    python train.py
    python test.py --quick
    python docs/generate_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))

import ml_pipeline as mlp  # noqa: E402

OUT_PDF = DOCS / "ML_Model_Implementation_Report.pdf"
FONT_DIR = DOCS / "fonts"


def register_fonts():
    """Use the bundled DejaVu fonts so mathematical symbols render properly.

    ReportLab's built-in Helvetica/Courier are limited to WinAnsi, which has no
    glyphs for the summation, square-root or sub/superscript characters used in
    the metric formulas - they would silently print as black boxes.  DejaVu is
    bundled under docs/fonts/ so the report regenerates identically on any
    machine; if the files are ever removed we fall back to the built-ins.
    """
    faces = {"Doc": "DejaVuSans.ttf", "Doc-Bold": "DejaVuSans-Bold.ttf",
             "Doc-Italic": "DejaVuSans-Oblique.ttf", "DocMono": "DejaVuSansMono.ttf"}
    try:
        for name, filename in faces.items():
            path = FONT_DIR / filename
            if not path.exists():
                raise FileNotFoundError(path)
            pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily("Doc", normal="Doc", bold="Doc-Bold",
                                      italic="Doc-Italic", boldItalic="Doc-Bold")
        return "Doc", "Doc-Bold", "Doc-Italic", "DocMono"
    except Exception as exc:
        print(f"  ! bundled fonts unavailable ({exc}); falling back to Helvetica")
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier"


BASE, BOLD, ITALIC, MONO = register_fonts()

INK = colors.HexColor("#12202f")
ACCENT = colors.HexColor("#0e7c5a")
ACCENT2 = colors.HexColor("#1f6feb")
MUTED = colors.HexColor("#54606f")
RULE = colors.HexColor("#c9d4e0")
BAND = colors.HexColor("#eef3f8")


# ---------------------------------------------------------------- styles
def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("t", parent=ss["Title"], fontName=BOLD,
                                fontSize=22, leading=27, textColor=INK, spaceAfter=4)
    s["sub"] = ParagraphStyle("sub", parent=ss["Normal"], fontName=BASE,
                              fontSize=11, leading=15, textColor=MUTED,
                              alignment=1, spaceAfter=2)
    s["h1"] = ParagraphStyle("h1", parent=ss["Heading1"], fontName=BOLD,
                             fontSize=14.5, leading=18, textColor=ACCENT,
                             spaceBefore=16, spaceAfter=7)
    s["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontName=BOLD,
                             fontSize=11.5, leading=15, textColor=INK,
                             spaceBefore=10, spaceAfter=5)
    s["body"] = ParagraphStyle("b", parent=ss["Normal"], fontName=BASE,
                               fontSize=9.7, leading=14.2, textColor=INK,
                               alignment=TA_JUSTIFY, spaceAfter=6)
    s["bullet"] = ParagraphStyle("bl", parent=s["body"], leftIndent=13,
                                 bulletIndent=4, spaceAfter=3)
    s["mono"] = ParagraphStyle("m", parent=ss["Normal"], fontName=MONO,
                               fontSize=8.6, leading=12.4,
                               textColor=colors.HexColor("#0b3550"),
                               backColor=colors.HexColor("#f2f6fa"),
                               borderPadding=6, leftIndent=3, spaceAfter=7)
    s["cell"] = ParagraphStyle("c", parent=ss["Normal"], fontName=BASE,
                               fontSize=8.1, leading=10.6, textColor=INK)
    s["cellb"] = ParagraphStyle("cb", parent=s["cell"], fontName=BOLD)
    s["note"] = ParagraphStyle("n", parent=s["body"], fontSize=8.8, leading=12.4,
                               textColor=MUTED)
    s["caption"] = ParagraphStyle("cap", parent=s["note"], spaceBefore=2,
                                  spaceAfter=10)
    return s


def para(text, st):
    return Paragraph(text, st)


def bullets(items, s, style="bullet"):
    return [Paragraph(f"&bull;&nbsp; {i}", s[style]) for i in items]


def table(data, s, widths, header=True, align=None, font_size=8.1,
          right_cols=()):
    """Build a table, wrapping every cell in a Paragraph.

    Raw strings inside a ReportLab Table are drawn as a single unbreakable
    line and ignore inline markup, so long cells overflow their column and
    tags such as <b> would be printed literally.  Converting each cell to a
    Paragraph makes text wrap inside the column and renders the markup.
    """
    cell_style = ParagraphStyle("tc", parent=s["cell"], fontSize=font_size,
                                leading=font_size * 1.32)
    head_style = ParagraphStyle("th", parent=cell_style, fontName=BOLD,
                                textColor=colors.white)
    right_style = ParagraphStyle("tr", parent=cell_style, alignment=TA_RIGHT)
    head_right = ParagraphStyle("thr", parent=head_style, alignment=TA_RIGHT)

    body = []
    for r, row in enumerate(data):
        out = []
        for c, value in enumerate(row):
            if isinstance(value, Paragraph):
                out.append(value)
                continue
            text = "" if value is None else str(value)
            if header and r == 0:
                style = head_right if c in right_cols else head_style
            else:
                style = right_style if c in right_cols else cell_style
            out.append(Paragraph(text, style))
        body.append(out)

    t = Table(body, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND])]
    for c in (align or []):
        cmds.append(c)
    t.setStyle(TableStyle(cmds))
    return t


def fmt(v, nd=4, dash="—"):
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return dash
    return f"{f:.{nd}f}"


# ---------------------------------------------------------------- content
def build(md: dict, bm: dict, s):
    story = []
    demand = md.get("demand", {}) or {}
    solar = md.get("solar", {}) or {}
    wind = md.get("wind", {}) or {}
    ds = md.get("dataset", {}) or {}
    split = md.get("split", {}) or {}
    tm = demand.get("test_metrics", {}) or {}
    prod = demand.get("production_model", "—")
    tol = bm.get("tolerance", mlp.DEFAULT_TOLERANCE)
    tol_pct = f"{tol * 100:.0f}%"
    results = [r for r in bm.get("results", []) if isinstance(r, dict)]
    thresholds = bm.get("demand_class_thresholds", {}) or {}

    # ---------------- cover ----------------
    story += [
        Spacer(1, 26 * mm),
        para("Machine Learning Model Implementation Report", s["title"]),
        para("Green Grid Allocator &mdash; Smart-Grid Demand and Renewable "
             "Forecasting", s["sub"]),
        para("Traditional machine learning on tabular hourly time-series data",
             s["sub"]),
        Spacer(1, 9 * mm),
    ]
    cover = [
        ["Production demand model", prod],
        ["Solar / wind models", f"{solar.get('production_model','—')} / "
                                f"{wind.get('production_model','—')}"],
        ["Forecast target", f"{demand.get('target','—')} ({demand.get('unit','—')})"],
        ["Dataset", f"{ds.get('file','—')} — {ds.get('hourly_rows','—')} hourly rows"],
        ["Features", f"{demand.get('n_features','—')} engineered features"],
        ["Held-out test R²", fmt(tm.get("r2"))],
        ["Held-out test RMSE", f"{fmt(tm.get('rmse'), 1)} kW"],
        [f"Accuracy within ±{tol_pct}", f"{fmt(tm.get('tolerance_accuracy'), 2)} %"],
        ["Model trained (UTC)", md.get("trained_at_utc", "—")],
        ["Benchmark run (UTC)", bm.get("generated_at_utc", "—")],
        ["Artifact version", md.get("artifact_version", "—")],
    ]
    story.append(table([[para(f"<b>{k}</b>", s["cell"]), para(str(v), s["cell"])]
                        for k, v in cover], s, [62 * mm, 96 * mm], header=False,
                       align=[("BACKGROUND", (0, 0), (0, -1), BAND)]))
    story += [
        Spacer(1, 8 * mm),
        para("<b>Scope statement.</b> This project uses traditional machine "
             "learning only. No deep-learning models (CNN, RNN, LSTM, GRU, "
             "transformers/BERT) and no image feature extractors (such as SIFT) "
             "are used anywhere, because the input data are numeric tabular "
             "time series rather than images or natural language. Section 5 "
             "explains this in detail.", s["note"]),
        PageBreak(),
    ]

    # ---------------- 1. purpose ----------------
    story += [
        para("1. Project Purpose and Forecasting Targets", s["h1"]),
        para("The Green Grid Allocator is an interactive smart-grid dispatch "
             "simulator. Each simulated hour it forecasts how much electricity "
             "the city will consume and how much renewable output is available, "
             "then dispatches the configured generation fleet in merit order "
             "(renewables first, then battery storage, then fossil plants by "
             "cost) to serve that demand with the least CO&#8322; possible.",
             s["body"]),
        para("Three quantities are predicted by machine learning:", s["body"]),
    ]
    story.append(table(
        [["#", "Target", "Type", "Model used", "Range / unit"],
         ["1", demand.get("target", "—"), "Regression (continuous)", prod, "kW"],
         ["2", "Solar capacity factor", "Regression (continuous)",
          solar.get("production_model", "—"), "0 – 1 (synthetic)"],
         ["3", "Wind capacity factor", "Regression (continuous)",
          wind.get("production_model", "—"), "0 – 1 (synthetic)"]],
        s, [10 * mm, 46 * mm, 36 * mm, 36 * mm, 30 * mm]))
    story += [
        para("The simulator converts the demand forecast from kW to MW and uses "
             "the two capacity factors to scale the installed solar and wind "
             "capacity into available MW.", s["caption"]),

        # ---------------- 2. dataset ----------------
        para("2. Input Dataset", s["h1"]),
        para(f"The primary dataset is <b>{ds.get('file','—')}</b>: power "
             "consumption and weather readings for Tetouan, Morocco, recorded "
             "every 10 minutes throughout 2017. After hourly resampling it "
             f"provides <b>{ds.get('hourly_rows','—')} rows</b> covering "
             f"{split.get('train_period',['—'])[0]} to "
             f"{split.get('test_period',['—','—'])[1]}.", s["body"]),
    ]
    story.append(table(
        [["Column", "Meaning", "Unit", "Role"],
         ["DateTime", "Timestamp of the reading", "—", "Index"],
         ["Temperature", "Ambient air temperature", "°C", "Feature"],
         ["Humidity", "Relative humidity", "%", "Feature"],
         ["Wind Speed", "Wind speed", "m/s", "Feature + wind target driver"],
         ["general diffuse flows", "Broad diffuse solar radiation", "W/m²",
          "Feature + solar target driver"],
         ["diffuse flows", "Diffuse solar radiation", "W/m²",
          "Feature + solar target driver"],
         ["Zone 1 Power Consumption", "Distribution zone 1 demand", "kW",
          "<b>Prediction target</b>"],
         ["Zone 2  Power Consumption", "Distribution zone 2 demand", "kW",
          "Lagged history only"],
         ["Zone 3  Power Consumption", "Distribution zone 3 demand", "kW",
          "Lagged history only"]],
        s, [42 * mm, 52 * mm, 16 * mm, 48 * mm]))
    story += [
        para("Zone 2 and Zone 3 readings for the current hour are deliberately "
             "excluded from the feature set — only their lagged and rolling "
             "history is used, since a live meter reading for the hour being "
             "predicted would not be available at forecast time.", s["caption"]),
        para("The repository also ships <i>Steel_industry_data.csv</i>, "
             "<i>smart_grid_stability_augmented.csv</i> and "
             "<i>household_power_consumption.txt</i>. These are retained from "
             "the original project for reference and are not used by the "
             "training, benchmarking or inference code.", s["note"]),
        PageBreak(),
        para("Derived target classes.", s["h2"]),
        para("For the auxiliary classification view (Section 8) the continuous "
             "demand is binned into three classes using tertiles of the "
             "training partition only:", s["body"]),
    ]
    story.append(table(
        [["Class", "Definition", "Boundary (kW)"],
         ["Low", "demand &lt; lower tertile", fmt(thresholds.get("low_medium"), 0)],
         ["Medium", "lower tertile ≤ demand &lt; upper tertile",
          f"{fmt(thresholds.get('low_medium'), 0)} – {fmt(thresholds.get('medium_high'), 0)}"],
         ["High", "demand ≥ upper tertile", fmt(thresholds.get("medium_high"), 0)]],
        s, [22 * mm, 84 * mm, 46 * mm]))

    # ---------------- 3. preprocessing ----------------
    story += [
        para("3. Data Preprocessing", s["h1"]),
        para("All preprocessing lives in <b>ml_pipeline.py</b> and is imported "
             "by <b>train.py</b>, <b>test.py</b> and <b>grid_sim.py</b>, so the "
             "feature definitions used at training time and at inference time "
             "can never drift apart.", s["body"]),
    ]
    story += bullets([
        "<b>Column normalisation</b> — surrounding whitespace is stripped from "
        "every CSV header.",
        "<b>Timestamp parsing</b> — <font face='DocMono'>DateTime</font> is "
        "parsed day-first (the file uses D/M/YYYY), set as the index and sorted "
        "chronologically.",
        "<b>Hourly resampling</b> — the six 10-minute readings in each hour are "
        "averaged to a regular 1-hour grid, which removes high-frequency meter "
        "noise and gives a uniform step for the lag features.",
        "<b>Missing values</b> — any hour with no reading is forward-filled "
        "(then back-filled at the very start), preserving the temporal order "
        "without inventing values from the future.",
        f"<b>Chronological split</b> — {int(mlp.SPLIT_FRACTIONS[0]*100)}% train / "
        f"{int(mlp.SPLIT_FRACTIONS[1]*100)}% validation / "
        f"{int(mlp.SPLIT_FRACTIONS[2]*100)}% test, cut by time. The data are "
        "<b>never shuffled</b>.",
        "<b>Scaling</b> — a <font face='DocMono'>StandardScaler</font> "
        "(zero mean, unit variance) is fitted on the training partition only "
        "and then applied to validation and test. For SVR and the MLP the "
        "target is standardised as well via "
        "<font face='DocMono'>TransformedTargetRegressor</font>, because "
        "demand values of ~3×10⁴ kW otherwise swamp their default loss scales.",
    ], s)
    story.append(table(
        [["Partition", "Rows", "Period", "Used for"],
         ["Train", str(split.get("n_train", "—")),
          " → ".join(split.get("train_period", ["—", "—"])), "Fitting every model"],
         ["Validation", str(split.get("n_val", "—")),
          " → ".join(split.get("val_period", ["—", "—"])),
          "Model selection + ensemble weights"],
         ["Test", str(split.get("n_test", "—")),
          " → ".join(split.get("test_period", ["—", "—"])),
          "Final reported metrics only"]],
        s, [24 * mm, 16 * mm, 68 * mm, 46 * mm]))
    story += [
        para("Because the split is chronological, the test period is a genuine "
             "future window relative to everything the models saw. This is "
             "stricter — and far more realistic for forecasting — than a random "
             "split, which would let a model interpolate between neighbouring "
             "hours it had already memorised.", s["caption"]),
        PageBreak(),
    ]

    # ---------------- 4. feature engineering ----------------
    story += [
        para("4. Feature Engineering", s["h1"]),
        para(f"The demand model consumes <b>{demand.get('n_features','—')} "
             "features</b> in four families. Every one is computed by "
             "<font face='DocMono'>ml_pipeline.build_demand_frame()</font>.",
             s["body"]),
    ]
    story.append(table(
        [["Family", "Features", "Count", "Why it helps"],
         ["Weather (current hour)",
          "Temperature, Humidity, Wind Speed, general diffuse flows, diffuse flows",
          "5", "Air-conditioning and lighting load track temperature and daylight."],
         ["Calendar + cyclical time",
          "hour, dow, month and sin/cos pairs for each",
          "9", "Encodes daily, weekly and seasonal rhythms; sin/cos keeps hour 23 "
               "adjacent to hour 0 instead of maximally distant."],
         ["Power lags &amp; rolling means",
          "For each of Zone 1/2/3: lag 24, lag 168, rolling mean 24, rolling mean 168",
          "12", "Yesterday's and last week's demand at the same hour are the single "
                "strongest predictors; rolling means capture the recent level."],
         ["Weather lags &amp; rolling means",
          "For each weather channel: lag 24, lag 168, rolling mean 24",
          "15", "Thermal inertia — buildings respond to the last day of weather, "
                "not only the current instant."]],
        s, [33 * mm, 50 * mm, 15 * mm, 60 * mm]))
    story += [
        para("Leakage prevention.", s["h2"]),
        para("Every historical feature used to predict hour <i>t</i> is built "
             "only from information available strictly before <i>t</i>:",
             s["body"]),
    ]
    story += bullets([
        "Lag features use <font face='DocMono'>shift(24)</font> and "
        "<font face='DocMono'>shift(168)</font>, exposing the value from exactly "
        "one day and one week earlier.",
        "Rolling means are computed and then <font face='DocMono'>shift(1)</font>-ed, "
        "so the window closes at <i>t</i>−1 and never includes the current hour.",
        "The raw Zone 1/2/3 readings at hour <i>t</i> are excluded from the "
        "feature list entirely.",
        "Scalers and class thresholds are fitted on training rows only.",
        "Current-hour weather <i>is</i> included, which is standard practice: at "
        "forecast time the weather for the target hour is a known exogenous "
        "input, supplied by a meteorological forecast.",
    ], s)
    story += [
        para("At inference time <font face='DocMono'>grid_sim.py</font> rebuilds "
             "exactly these features from the real historical series for the "
             "simulator's current hour, using the same window semantics, and "
             "reads the ordered feature list from "
             "<font face='DocMono'>models/features.json</font> so the vector is "
             "assembled in the order the model was trained on.", s["body"]),

        # ---------------- 5. SIFT ----------------
        para("5. Why SIFT and Deep Learning Are Not Used", s["h1"]),
        para("<b>SIFT (Scale-Invariant Feature Transform) is not used in this "
             "project, and it would not be meaningful to use it.</b> SIFT is a "
             "computer-vision algorithm that locates keypoints in a 2-D image "
             "and describes the gradient patterns around them. It presupposes "
             "pixels, spatial neighbourhoods and scale space. The inputs here "
             "are numeric readings from meters and weather sensors arranged as "
             "rows in a table — there is no image, no pixel grid and no spatial "
             "structure for a keypoint detector to operate on. The same applies "
             "to any CNN-based image feature extraction.", s["body"]),
        para("For the same category reason, this project also does not use "
             "text transformers. BERT and related models operate on tokenised "
             "natural language; a sequence of kilowatt readings is not text, so "
             "reporting a BERT score here would be meaningless. The benchmark "
             "therefore prints an explicit <i>not applicable</i> row for these "
             "families rather than fabricating numbers for them.", s["body"]),
        para("Deep sequence models (RNN, LSTM, GRU) <i>are</i> technically "
             "applicable to time series, but they are outside this project's "
             "scope: the requirement is a traditional machine-learning "
             "implementation. No TensorFlow or PyTorch dependency is present. "
             "The feature engineering in Section 4 is what supplies the "
             "temporal context instead — the lag and rolling-window features "
             "hand the model an explicit summary of the previous day and week, "
             "which is the role a recurrent layer would otherwise play.",
             s["body"]),
        para("<b>The actual feature extraction used is therefore:</b> "
             "time-derived cyclical encodings, lag values, rolling statistics, "
             "and weather variables — all tabular, all numeric.", s["body"]),
        PageBreak(),
    ]

    # ---------------- 6. algorithms ----------------
    story += [
        para("6. Algorithms Benchmarked", s["h1"]),
        para("Every model below is trained on the same training partition and "
             "scored on the same held-out test period. All are traditional "
             "machine-learning algorithms from scikit-learn, plus XGBoost.",
             s["body"]),
    ]
    algo_rows = [["Algorithm", "Family", "Role and suitability for this task"]]
    algo_desc = [
        ("Linear Regression", "Linear",
         "Ordinary least squares. Transparent baseline; underfits because load "
         "responds non-linearly to temperature and hour."),
        ("Ridge Regression", "Linear (L2)",
         "Adds L2 shrinkage, which stabilises coefficients when the lag and "
         "rolling features are strongly correlated with each other."),
        ("Lasso Regression", "Linear (L1)",
         "L1 penalty drives weak coefficients to zero, giving implicit feature "
         "selection over the 41 inputs."),
        ("ElasticNet", "Linear (L1+L2)",
         "Blends both penalties; handles correlated feature groups better than "
         "pure Lasso."),
        ("KNN Regressor", "Instance-based",
         "Averages the nearest historical hours in feature space. Intuitive, "
         "but weakens in high dimensions."),
        ("Decision Tree", "Tree",
         "Single interpretable tree; captures non-linear thresholds but has "
         "high variance on its own."),
        ("Random Forest", "Bagged trees",
         "Averages many decorrelated deep trees. Robust, handles feature "
         "interactions, needs little tuning."),
        ("Extra Trees", "Bagged trees",
         "Like Random Forest but with randomised split thresholds — lower "
         "variance and noticeably faster to fit."),
        ("Gradient Boosting", "Boosted trees",
         "Fits trees sequentially on residuals. Strong on structured tabular "
         "data; slower to train."),
        ("HistGradientBoosting", "Boosted trees",
         "Histogram-binned boosting — near-identical accuracy to classic "
         "boosting at a fraction of the training time."),
        ("AdaBoost", "Boosted trees",
         "Re-weights hard examples across rounds, using depth-8 trees as base "
         "learners here."),
        ("Support Vector Regressor", "Kernel",
         "RBF-kernel epsilon-tube regression. Applicable, but scales poorly "
         "with sample count and is sensitive to feature/target scaling."),
        ("MLP Regressor", "Shallow neural net",
         "scikit-learn's fully-connected network. Included as the traditional "
         "ML upper bound; it is not a deep-learning framework model."),
        ("XGBoost", "Boosted trees",
         "Regularised gradient boosting; a standard strong baseline for tabular "
         "regression. Optional — the benchmark skips it cleanly if not "
         "installed."),
        ("Gaussian Naive Bayes", "Probabilistic classifier",
         "<b>Classification baseline only.</b> Predicts a Low/Medium/High "
         "demand class, never a continuous value — see Section 7."),
    ]
    for name, fam, why in algo_desc:
        algo_rows.append([para(f"<b>{name}</b>", s["cell"]), para(fam, s["cell"]),
                          para(why, s["cell"])])
    story.append(table(algo_rows, s, [36 * mm, 26 * mm, 96 * mm]))

    story += [
        para("Explicitly not applicable.", s["h2"]),
    ]
    na_rows = [["Method", "Why it is not applicable here"]]
    for k, v in (bm.get("not_applicable") or mlp.NOT_APPLICABLE_MODELS).items():
        na_rows.append([para(f"<b>{k}</b>", s["cell"]), para(v, s["cell"])])
    story.append(table(na_rows, s, [40 * mm, 118 * mm]))

    # ---------------- 7. regression vs NB ----------------
    story += [
        para("7. Regression Models vs. the Naive Bayes Baseline", s["h1"]),
        para("This distinction matters because it is a common source of "
             "misreported results. Demand forecasting is a <b>regression</b> "
             "problem: the answer is a real number of kilowatts. Naive Bayes is "
             "a <b>classifier</b>: it outputs a discrete label and a class "
             "probability, and it has no mechanism for producing a continuous "
             "kilowatt value.", s["body"]),
        para("Therefore GaussianNB is never treated as a demand regressor in "
             "this project. It is trained on the derived Low/Medium/High labels "
             "and reported separately, with MAE, RMSE and R² left blank because "
             "those quantities are undefined for it. Conversely, the regression "
             "models are never presented as classifiers — their precision, "
             "recall and F1 are always labelled as <i>derived</i> demand-class "
             "metrics obtained by binning their continuous output.", s["body"]),

        # ---------------- 8. metrics ----------------
        para("8. Evaluation Metrics", s["h1"]),
        para("Native regression metrics.", s["h2"]),
    ]
    story.append(table(
        [["Metric", "Formula", "Reading"],
         ["MAE", "(1/n) · Σ |yᵢ − ŷᵢ|",
          "Mean absolute error in kW. Same units as demand; robust to outliers."],
         ["RMSE", "√[ (1/n) · Σ (yᵢ − ŷᵢ)² ]",
          "Root mean squared error in kW. Penalises large misses more heavily."],
         ["R²", "1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²",
          "Share of demand variance explained. 1.0 is perfect; 0.0 equals always "
          "predicting the mean."],
         [f"Accuracy within ±{tol_pct}",
          "100 · (1/n) · Σ 1[ |ŷᵢ − yᵢ| ≤ τ·|yᵢ| ]",
          f"Percentage of hours forecast within ±{tol_pct} of actual demand, with "
          f"τ = {tol}. Configurable via --tolerance."]],
        s, [30 * mm, 52 * mm, 76 * mm]))
    story += [
        para("Derived demand-class metrics.", s["h2"]),
        para("Precision, recall, F1 and classification accuracy do not exist "
             "natively for a regression task. They are obtained here through a "
             "documented auxiliary view, and are always labelled as derived:",
             s["body"]),
    ]
    story += bullets([
        "Compute tertile thresholds on the <b>training</b> target only.",
        "Apply those exact thresholds to both the actual and the predicted test "
        "demand, producing Low/Medium/High labels for each.",
        "Compute <b>macro-averaged</b> precision, recall and F1 across the three "
        "classes, so each class counts equally regardless of its frequency.",
    ], s)
    story.append(table(
        [["Metric", "Formula", "Reading"],
         ["Precision", "TP / (TP + FP)",
          "Of the hours predicted to be in a class, the share that truly were."],
         ["Recall", "TP / (TP + FN)",
          "Of the hours truly in a class, the share that were found."],
         ["F1", "2 · (P · R) / (P + R)",
          "Harmonic mean of precision and recall."],
         ["Class accuracy", "correct labels / total labels",
          "Share of test hours placed in the correct demand band."]],
        s, [30 * mm, 42 * mm, 86 * mm]))
    story += [
        para("A model can post a high derived F1 while still having mediocre "
             "RMSE — landing in the right third of the demand range is a much "
             "easier task than predicting the exact kilowatts. This is precisely "
             "why these figures must never be quoted as regression accuracy.",
             s["caption"]),
        PageBreak(),
    ]

    # ---------------- 9. results ----------------
    story += [
        para("9. Benchmark Results", s["h1"]),
        para(f"Produced by <font face='DocMono'>python test.py --quick</font> on "
             f"{bm.get('generated_at_utc','—')} (seed {bm.get('seed','—')}), "
             f"evaluated on the held-out test period "
             f"{' → '.join(split.get('test_period', ['—','—']))}. Derived "
             "columns are marked with an asterisk.", s["body"]),
    ]
    hdr = ["#", "Model", "Type", f"±{tol_pct} Acc", "MAE", "RMSE", "R²",
           "Prec*", "Rec*", "F1*"]
    rows = [hdr]
    ranked = sorted([r for r in results if r.get("rank")], key=lambda r: r["rank"])
    others = [r for r in results if not r.get("rank")]
    for r in ranked + others:
        rows.append([
            str(r.get("rank") or "—"),
            para(f"<b>{r.get('model','')}</b>" if r.get("rank") in (1, 2)
                 else r.get("model", ""), s["cell"]),
            para(r.get("type", ""), s["cell"]),
            fmt(r.get("tolerance_accuracy"), 2),
            fmt(r.get("mae"), 1), fmt(r.get("rmse"), 1), fmt(r.get("r2"), 4),
            fmt(r.get("derived_precision"), 3),
            fmt(r.get("derived_recall"), 3),
            fmt(r.get("derived_f1"), 3),
        ])
    story.append(table(rows, s, [7 * mm, 40 * mm, 21 * mm, 15 * mm, 15 * mm,
                                 15 * mm, 14 * mm, 12 * mm, 12 * mm, 12 * mm],
                       font_size=7.3, right_cols=(3, 4, 5, 6, 7, 8, 9)))
    story += [
        para("* Derived demand-class metrics (Section 8), not native regression "
             "scores. Rows without a rank do not predict continuous demand and "
             "are excluded from ranking: the Naive Bayes classification baseline "
             "and the not-applicable method families.", s["caption"]),

        # ---------------- 10. ensembles ----------------
        para("10. Model Combinations (Ensembles)", s["h1"]),
        para("Three genuine weighted ensembles are built. A weighted ensemble "
             "produces its forecast as a convex blend of its members' "
             "predictions:", s["body"]),
        para("ŷ<sub>ens</sub> = w₁·ŷ₁ + w₂·ŷ₂ + … + w<sub>k</sub>·ŷ<sub>k</sub>"
             "&nbsp;&nbsp; subject to&nbsp; w<sub>i</sub> ≥ 0&nbsp; and&nbsp; "
             "Σ w<sub>i</sub> = 1", s["mono"]),
        para("The weights are fitted on the <b>validation</b> partition only, by "
             "minimising validation RMSE over the simplex (SLSQP, with "
             "non-negative least squares as a fallback). Constraining weights to "
             "be non-negative and to sum to one keeps the blend interpretable "
             "and prevents one member's errors being amplified to cancel "
             "another's. The resulting weights are then applied unchanged to the "
             "untouched test predictions.", s["body"]),
        para("Ensembles help when members make <i>different</i> mistakes: "
             "averaging cancels independent error while preserving the shared "
             "signal. When members are highly correlated, the optimiser can "
             "legitimately place all weight on the single best member — the "
             "benchmark reports that honestly rather than forcing an artificial "
             "blend.", s["body"]),
    ]
    ens_rows = [["Ensemble", "Fitted weights (validation)", f"±{tol_pct} Acc", "RMSE", "R²"]]
    for r in results:
        if r.get("type") == "ensemble":
            ens_rows.append([
                para(r.get("model", ""), s["cell"]),
                para(r.get("status", ""), s["cell"]),
                fmt(r.get("tolerance_accuracy"), 2),
                fmt(r.get("rmse"), 1), fmt(r.get("r2"), 4)])
    if len(ens_rows) > 1:
        story.append(table(ens_rows, s, [42 * mm, 62 * mm, 18 * mm, 18 * mm, 18 * mm],
                           font_size=7.3, right_cols=(2, 3, 4)))

    # ---------------- 11. selection ----------------
    story += [
        para("11. How the Best Model Is Selected", s["h1"]),
        para(f"All models and ensembles are ranked by a single declared rule: "
             f"<b>{bm.get('ranking_rule', mlp.RANKING_RULE_TEXT)}</b>.",
             s["body"]),
    ]
    story += bullets([
        f"<b>Primary — accuracy within ±{tol_pct}.</b> This is the criterion an "
        "operator cares about: how often the forecast is close enough to be "
        "actionable.",
        "<b>First tie-break — lowest RMSE.</b> Among models that hit the band "
        "equally often, prefer the one whose misses are smaller.",
        "<b>Second tie-break — highest R².</b> Prefer the model explaining more "
        "of the variance overall.",
    ], s)
    bi, be = bm.get("best_individual"), bm.get("best_ensemble")
    sel = [["Category", "Winner", "Why"]]
    if bi:
        sel.append([para("<b>Best individual</b>", s["cell"]),
                    para(bi.get("model", "—"), s["cell"]),
                    para(bm.get("best_individual_reason", "—"), s["cell"])])
    if be:
        sel.append([para("<b>Best combination</b>", s["cell"]),
                    para(be.get("model", "—"), s["cell"]),
                    para(bm.get("best_ensemble_reason", "—"), s["cell"])])
    sel.append([para("<b>Production model</b>", s["cell"]),
                para(prod, s["cell"]),
                para("Selected by <b>train.py</b> using the same rule but scored "
                     "on the <b>validation</b> partition, because choosing a "
                     "production model on test results would leak the test set "
                     "and inflate the reported performance. The production "
                     "choice may therefore differ from the test-set winner — "
                     "that difference is expected and is the honest outcome of "
                     "keeping the test period untouched.", s["cell"])])
    story.append(table(sel, s, [30 * mm, 34 * mm, 94 * mm]))
    story += [
        para(f"Held-out test performance of the deployed model ({prod}): "
             f"MAE {fmt(tm.get('mae'), 1)} kW, RMSE {fmt(tm.get('rmse'), 1)} kW, "
             f"R² {fmt(tm.get('r2'))}, accuracy within ±{tol_pct} "
             f"{fmt(tm.get('tolerance_accuracy'), 2)}%, derived macro-F1 "
             f"{fmt(tm.get('derived_f1'))}.", s["caption"]),
        PageBreak(),
    ]

    # ---------------- 12. production placement + outputs ----------------
    story += [
        para("12. Production Deployment and Outputs", s["h1"]),
        para("Where each model lives in the running application:", s["body"]),
    ]
    story.append(table(
        [["Stage", "File", "What happens"],
         ["Training", "train.py",
          "Fits every candidate, selects on validation, refits the winner on "
          "train+validation, writes artifacts to models/."],
         ["Shared pipeline", "ml_pipeline.py",
          "Single definition of cleaning, features, splitting, metrics and the "
          "model registry."],
         ["Benchmark", "test.py",
          "Trains and scores the full comparison, writes benchmark_results.json "
          "and .csv."],
         ["Inference", "grid_sim.py",
          "GridSimulator.load_artifacts() loads the production models; "
          "predict_demand(), predict_solar_cf() and predict_wind_cf() run each "
          "simulated hour."],
         ["Web API", "app.py",
          "Serves the simulator endpoints plus GET /api/model-performance, which "
          "feeds the dashboard's Model Performance panel."]],
        s, [28 * mm, 30 * mm, 100 * mm]))
    story += [para("Output formats.", s["h2"])]
    story.append(table(
        [["Output", "Type", "Produced by", "Consumed as"],
         ["Demand forecast", "Continuous float, kW", f"{prod} regressor",
          "Divided by 1000 → MW; drives the dispatch stack."],
         ["Solar capacity factor", "Continuous float, 0–1",
          f"{solar.get('production_model','—')} regressor",
          "Clipped to [0,1]; multiplied by installed solar MW."],
         ["Wind capacity factor", "Continuous float, 0–1",
          f"{wind.get('production_model','—')} regressor",
          "Clipped to [0,1]; multiplied by installed wind MW."],
         ["Low / Medium / High class", "Discrete label",
          "Derived binning (and GaussianNB baseline)",
          "Evaluation and reporting only — the simulator itself always uses the "
          "continuous forecast."]],
        s, [32 * mm, 28 * mm, 40 * mm, 58 * mm]))
    story += [
        para("The demand-class output is a reporting construct. It is not fed "
             "into dispatch, because dispatch needs an actual megawatt figure "
             "rather than a band.", s["caption"]),
    ]

    # ---------------- 13. commands ----------------
    story += [
        para("13. Installation and Execution", s["h1"]),
        para("<b>1. Install dependencies</b>", s["h2"]),
        para("cd mlpbl<br/>python -m venv .venv<br/>"
             "source .venv/bin/activate&nbsp;&nbsp;# Windows: .venv\\Scripts\\activate<br/>"
             "pip install -r requirements.txt", s["mono"]),
        para("<b>2. Train the production models</b> (writes to models/)", s["h2"]),
        para("python train.py<br/>"
             "python train.py --full&nbsp;&nbsp;&nbsp;&nbsp;# heavier hyper-parameters<br/>"
             "python train.py --tolerance 0.05", s["mono"]),
        para("<b>3. Run the benchmark</b> (writes benchmark_results.json / .csv)",
             s["h2"]),
        para("python test.py<br/>"
             "python test.py --quick<br/>"
             "python test.py --full<br/>"
             "python test.py --tolerance 0.10", s["mono"]),
        para("<b>4. Launch the interfaces</b>", s["h2"]),
        para("python app.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# web UI → http://127.0.0.1:8000<br/>"
             "python grid_sim.py&nbsp;# terminal simulator", s["mono"]),
        para("<b>5. Regenerate this report</b>", s["h2"]),
        para("python docs/generate_report.py", s["mono"]),
        para("Training must run before the app: the simulator raises an "
             "actionable error naming the missing artifacts and the exact "
             "command to fix it, rather than failing with a raw traceback.",
             s["note"]),

        # ---------------- 14. limitations ----------------
        para("14. Limitations", s["h1"]),
    ]
    story += bullets([
        "<b>Solar and wind targets are synthetic.</b> The dataset contains no "
        "measured renewable generation. Solar capacity factor is derived from "
        "diffuse-flow readings plus a solar-altitude and seasonality model; wind "
        "capacity factor comes from a standard turbine power curve applied to "
        "wind speed. Their very high R² values therefore mainly show that the "
        "models learned a deterministic formula — they are <i>not</i> evidence "
        "of real-world renewable forecast skill, and should not be quoted as "
        "such. Only the demand metrics reflect a genuine prediction problem.",
        "<b>Classification metrics are derived, not native.</b> Precision, "
        "recall and F1 come from binning a continuous forecast into tertiles. "
        "They measure band placement, not kilowatt accuracy, and are easier to "
        "score well on.",
        "<b>Single city, single year.</b> The data cover Tetouan in 2017. "
        "Seasonal coverage is complete but there is only one instance of each "
        "season, so the model cannot distinguish a recurring seasonal pattern "
        "from a one-off event that year.",
        "<b>The test period is winter.</b> A chronological split puts "
        "November–December in the test set, so reported metrics reflect "
        "cold-season load behaviour more than summer cooling peaks.",
        "<b>One-step-ahead framing.</b> The model predicts the current hour "
        "given weather and history up to the previous hour. Multi-day-ahead "
        "forecasting would require recursive prediction and would show "
        "materially worse error.",
        "<b>Hardware and runtime.</b> Default (quick) hyper-parameters are sized "
        "to finish in a couple of minutes on a normal laptop CPU. --full "
        "improves some models but takes considerably longer; no GPU is used or "
        "needed anywhere.",
        "<b>Weather is assumed known.</b> Current-hour weather is treated as an "
        "available input. In live operation it would come from a meteorological "
        "forecast, whose own error would add to the demand forecast error.",
    ], s)

    story += [
        para("15. Project File Map", s["h1"]),
    ]
    story.append(table(
        [["Path", "Purpose"],
         ["ml_pipeline.py", "Shared preprocessing, features, splits, metrics, model registry"],
         ["train.py", "Trains and saves the production models"],
         ["test.py", "Full benchmark + ensembles → JSON/CSV"],
         ["grid_sim.py", "Simulator core and terminal UI; production inference"],
         ["app.py", "FastAPI server and REST API"],
         ["static/", "Dashboard (index.html, app.js, style.css)"],
         ["models/", "Trained artifacts, metadata and benchmark results"],
         ["docs/", "This report and its generator script"],
         ["Tetuan City power consumption.csv", "Primary dataset"]],
        s, [56 * mm, 102 * mm]))

    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont(BASE, 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9.5 * mm,
                      "Green Grid Allocator — ML Model Implementation Report")
    canvas.drawRightString(A4[0] - 18 * mm, 9.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


OUT_MD = DOCS / "ML_Model_Implementation_Report.md"


def write_markdown(md: dict, bm: dict) -> None:
    """Write a markdown companion carrying the same real numbers as the PDF."""
    demand = md.get("demand", {}) or {}
    solar = md.get("solar", {}) or {}
    wind = md.get("wind", {}) or {}
    split = md.get("split", {}) or {}
    ds = md.get("dataset", {}) or {}
    tm = demand.get("test_metrics", {}) or {}
    th = bm.get("demand_class_thresholds", {}) or {}
    tol = bm.get("tolerance", mlp.DEFAULT_TOLERANCE)
    tol_pct = f"{tol * 100:.0f}%"
    results = [r for r in bm.get("results", []) if isinstance(r, dict)]

    L = []
    A = L.append
    A("# ML Model Implementation Report — Green Grid Allocator\n")
    A("> **Source note.** The authoritative, editable source for the PDF is\n"
      "> `docs/generate_report.py` — all prose lives there and every figure is\n"
      "> read from `models/model_metadata.json` and `models/benchmark_results.json`.\n"
      "> This markdown file is generated by the same script, so the two can never\n"
      "> disagree. Regenerate both with `python docs/generate_report.py`.\n")
    A("## Summary\n")
    A(f"| Item | Value |\n|---|---|")
    A(f"| Production demand model | {demand.get('production_model','—')} |")
    A(f"| Solar model | {solar.get('production_model','—')} |")
    A(f"| Wind model | {wind.get('production_model','—')} |")
    A(f"| Target | {demand.get('target','—')} ({demand.get('unit','—')}) |")
    A(f"| Dataset | {ds.get('file','—')} — {ds.get('hourly_rows','—')} hourly rows |")
    A(f"| Features | {demand.get('n_features','—')} |")
    A(f"| Test MAE | {fmt(tm.get('mae'), 1)} kW |")
    A(f"| Test RMSE | {fmt(tm.get('rmse'), 1)} kW |")
    A(f"| Test R² | {fmt(tm.get('r2'))} |")
    A(f"| Accuracy within ±{tol_pct} | {fmt(tm.get('tolerance_accuracy'), 2)} % |")
    A(f"| Trained (UTC) | {md.get('trained_at_utc','—')} |")
    A(f"| Benchmark (UTC) | {bm.get('generated_at_utc','—')} |\n")

    A("## Approach\n")
    A("Traditional machine learning only. No deep-learning frameworks "
      "(TensorFlow/PyTorch) and no CNN, RNN, LSTM, GRU or transformer models are "
      "used. **SIFT is not used and is not applicable**: it is an image keypoint "
      "descriptor, while this project's inputs are numeric tabular time series. "
      "The actual feature extraction is: weather variables, cyclical time "
      "encodings, lag values (24 h, 168 h) and rolling statistics (24 h, 168 h).\n")

    A("## Chronological split (never shuffled)\n")
    A("| Partition | Rows | Period | Used for |\n|---|---|---|---|")
    for key, label, use in [("train", "Train", "Fitting every model"),
                            ("val", "Validation", "Model selection + ensemble weights"),
                            ("test", "Test", "Final reported metrics only")]:
        per = split.get(f"{key}_period", ["—", "—"])
        A(f"| {label} | {split.get(f'n_{key}','—')} | {per[0]} → {per[1]} | {use} |")
    A("")
    A(f"Demand classes (training tertiles): Low < {fmt(th.get('low_medium'), 0)} kW "
      f"≤ Medium < {fmt(th.get('medium_high'), 0)} kW ≤ High\n")

    A("## Benchmark results\n")
    A(f"Ranking rule: **{bm.get('ranking_rule', mlp.RANKING_RULE_TEXT)}**. "
      "Columns marked \\* are *derived demand-class metrics*, obtained by binning "
      "actual and predicted demand into Low/Medium/High — they are not native "
      "regression scores.\n")
    A(f"| # | Model | Type | ±{tol_pct} Acc | MAE | RMSE | R² | Prec\\* | Rec\\* | F1\\* |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    ranked = sorted([r for r in results if r.get("rank")], key=lambda r: r["rank"])
    for r in ranked + [r for r in results if not r.get("rank")]:
        A(f"| {r.get('rank') or '—'} | {r.get('model','')} | {r.get('type','')} | "
          f"{fmt(r.get('tolerance_accuracy'), 2)} | {fmt(r.get('mae'), 1)} | "
          f"{fmt(r.get('rmse'), 1)} | {fmt(r.get('r2'))} | "
          f"{fmt(r.get('derived_precision'), 3)} | {fmt(r.get('derived_recall'), 3)} | "
          f"{fmt(r.get('derived_f1'), 3)} |")
    A("")
    if bm.get("best_individual"):
        A(f"**Best individual model — {bm['best_individual'].get('model')}**  \n"
          f"{bm.get('best_individual_reason','')}\n")
    if bm.get("best_ensemble"):
        A(f"**Best combination — {bm['best_ensemble'].get('model')}**  \n"
          f"{bm.get('best_ensemble_reason','')}  \n"
          f"Weights: {bm['best_ensemble'].get('status','')}\n")
    A(f"**Production model — {demand.get('production_model','—')}**  \n"
      "Chosen by `train.py` using the same ranking rule but scored on the "
      "**validation** partition. Selecting on test results would leak the test "
      "set, so the production model may differ from the test-set winner.\n")

    A("## Not applicable\n")
    A("| Method | Why |\n|---|---|")
    for k, v in (bm.get("not_applicable") or mlp.NOT_APPLICABLE_MODELS).items():
        A(f"| {k} | {v} |")
    A("")
    A("## Commands\n")
    A("```bash\npip install -r requirements.txt\npython train.py\n"
      "python test.py --quick\npython app.py\npython docs/generate_report.py\n```\n")
    A("Full detail — preprocessing, feature families, metric formulas, ensemble "
      "mathematics, deployment map and limitations — is in "
      "`ML_Model_Implementation_Report.pdf`.\n")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


def main():
    md = mlp.read_json_safe(mlp.METADATA_PATH)
    bm = mlp.read_json_safe(mlp.BENCHMARK_JSON)
    if not md:
        raise SystemExit("models/model_metadata.json not found — run `python train.py` first.")
    if not bm:
        raise SystemExit("models/benchmark_results.json not found — run `python test.py` first.")

    s = build_styles()
    DOCS.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title="ML Model Implementation Report — Green Grid Allocator",
        author="Green Grid Allocator", subject="Traditional ML implementation")
    doc.build(build(md, bm, s), onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT_PDF}  ({OUT_PDF.stat().st_size / 1024:.0f} KB)")
    write_markdown(md, bm)


if __name__ == "__main__":
    main()
