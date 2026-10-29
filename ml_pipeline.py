"""Shared ML pipeline for the Green Grid Allocator.

Only these individual regression algorithms are allowed:
- KNN Regressor
- Random Forest
- Gradient Boosting
- XGBoost
- Linear Regression

`train.py`, `test.py`, and `grid_sim.py` import this file.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==========================================================================
# Paths and constants
# ==========================================================================
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "Tetuan City power consumption.csv"

BENCHMARK_JSON = MODEL_DIR / "benchmark_results.json"
BENCHMARK_CSV = MODEL_DIR / "benchmark_results.csv"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

ARTIFACT_VERSION = "3.0-approved-models-only"
RANDOM_SEED = 42

TARGET_DEMAND = "Zone 1 Power Consumption"

ZONE_COLS = [
    "Zone 1 Power Consumption",
    "Zone 2  Power Consumption",
    "Zone 3  Power Consumption",
]

WEATHER_COLS = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
]

LAGS = (24, 168)
ROLLING_WINDOWS = (24, 168)
WEATHER_ROLLING_WINDOW = 24

# 70% training, 15% validation, 15% final untouched test set.
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

DEFAULT_TOLERANCE = 0.10
DEMAND_CLASS_NAMES = ["Low", "Medium", "High"]

RANKING_RULE_TEXT = (
    "highest tolerance accuracy, then lowest RMSE, then highest R²"
)

PRODUCTION_FILES = [
    "demand_model.pkl",
    "solar_model.pkl",
    "wind_model.pkl",
    "scaler_demand.pkl",
    "scaler_solar.pkl",
    "scaler_wind.pkl",
    "features.json",
    "last_weather.json",
]


# ==========================================================================
# Dataset loading and cleaning
# ==========================================================================
def load_and_clean(data_path: Path | str | None = None) -> pd.DataFrame:
    """Load the CSV, parse timestamps, resample to hourly data, and fill gaps."""
    path = Path(data_path) if data_path else DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place 'Tetuan City power consumption.csv' in the project folder."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "DateTime" not in df.columns:
        raise KeyError("The dataset must contain a DateTime column.")

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        dayfirst=True,
        format="mixed",
    )

    df = df.set_index("DateTime").sort_index()
    df = df.resample("1h").mean()
    df = df.ffill().bfill()

    return df


# ==========================================================================
# Feature engineering
# ==========================================================================
def time_features(timestamp) -> dict:
    """Return calendar and cyclical time features for one timestamp."""
    hour = timestamp.hour
    dow = timestamp.weekday()
    month = timestamp.month

    return {
        "hour": hour,
        "dow": dow,
        "month": month,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time and cyclical time features."""
    out = df.copy()

    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out


def add_lag_rolling(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Add leakage-safe lag and rolling features for a power column."""
    out = df.copy()

    for lag in LAGS:
        out[f"{column}_lag{lag}"] = out[column].shift(lag)

    for window in ROLLING_WINDOWS:
        out[f"{column}_roll{window}"] = (
            out[column]
            .rolling(window, min_periods=1)
            .mean()
            .shift(1)
        )

    return out


def add_weather_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe weather lag and rolling features."""
    out = df.copy()

    for column in WEATHER_COLS:
        for lag in LAGS:
            out[f"{column}_lag{lag}"] = out[column].shift(lag)

        out[f"{column}_roll{WEATHER_ROLLING_WINDOW}"] = (
            out[column]
            .rolling(WEATHER_ROLLING_WINDOW, min_periods=1)
            .mean()
            .shift(1)
        )

    return out


def demand_feature_names() -> list[str]:
    """Return the exact ordered feature names used by demand models."""
    features = list(WEATHER_COLS)

    features += [
        "hour",
        "dow",
        "month",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]

    for column in ZONE_COLS:
        for lag in LAGS:
            features.append(f"{column}_lag{lag}")

        for window in ROLLING_WINDOWS:
            features.append(f"{column}_roll{window}")

    for column in WEATHER_COLS:
        for lag in LAGS:
            features.append(f"{column}_lag{lag}")

        features.append(f"{column}_roll{WEATHER_ROLLING_WINDOW}")

    return features


def build_demand_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build the demand model frame using weather, time, lag, and rolling data."""
    out = add_time_features(df)

    for column in ZONE_COLS:
        out = add_lag_rolling(out, column)

    out = add_weather_lags(out)
    out = out.dropna()

    features = demand_feature_names()
    missing = [feature for feature in features if feature not in out.columns]

    if missing:
        raise KeyError(f"Missing engineered features: {missing}")

    return out, features


# ==========================================================================
# Solar and wind target creation
# ==========================================================================
def make_solar_target(df: pd.DataFrame) -> np.ndarray:
    """Create synthetic solar capacity factor from weather and calendar data."""
    hour = df.index.hour.values
    month = df.index.month.values

    diffuse = df["diffuse flows"].values
    general_diffuse = df["general diffuse flows"].values
    temperature = df["Temperature"].values

    solar_altitude = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    seasonal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (month - 3) / 12)

    cloud_factor = 1 - (
        general_diffuse / (general_diffuse + diffuse + 1e-6)
    )

    temperature_derate = np.where(
        temperature > 25,
        1 - 0.005 * (temperature - 25),
        1.0,
    )

    capacity_factor = (
        solar_altitude
        * seasonal_factor
        * (0.3 + 0.7 * cloud_factor)
        * temperature_derate
    )

    return np.clip(capacity_factor, 0, 1)


def make_wind_target(df: pd.DataFrame) -> np.ndarray:
    """Create synthetic wind capacity factor using a turbine power curve."""
    wind_speed = df["Wind Speed"].values
    capacity_factor = np.zeros_like(wind_speed, dtype=float)

    operating = (wind_speed >= 3) & (wind_speed < 12)
    capacity_factor[operating] = (
        (wind_speed[operating] - 3) / 9
    ) ** 3

    capacity_factor[wind_speed >= 12] = 1.0
    capacity_factor[wind_speed >= 25] = 0.0

    return capacity_factor


SOLAR_FEATURES = [
    "Temperature",
    "Humidity",
    "diffuse flows",
    "general diffuse flows",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

WIND_FEATURES = [
    "Wind Speed",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def build_solar_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build solar features and synthetic solar capacity-factor target."""
    out = add_time_features(df)
    out["solar_cf"] = make_solar_target(out)
    return out.dropna(), list(SOLAR_FEATURES)


def build_wind_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build wind features and synthetic wind capacity-factor target."""
    out = add_time_features(df)
    out["wind_cf"] = make_wind_target(out)
    return out.dropna(), list(WIND_FEATURES)


# ==========================================================================
# Chronological splitting
# ==========================================================================
def split_boundaries(
    index: pd.DatetimeIndex,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
):
    """Return validation and test boundary timestamps."""
    count = len(index)

    validation_start_index = int(count * fractions[0])
    test_start_index = int(count * (fractions[0] + fractions[1]))

    return index[validation_start_index], index[test_start_index]


def split_by_time(
    frame: pd.DataFrame,
    validation_start,
    test_start,
):
    """Split a frame chronologically into training, validation, and test."""
    train = frame[frame.index < validation_start]
    validation = frame[
        (frame.index >= validation_start)
        & (frame.index < test_start)
    ]
    test = frame[frame.index >= test_start]

    return train, validation, test


@dataclass
class DemandSplit:
    """Prepared chronological demand train/validation/test partitions."""

    feature_names: list[str]
    val_start: pd.Timestamp
    test_start: pd.Timestamp

    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    index_train: pd.DatetimeIndex
    index_val: pd.DatetimeIndex
    index_test: pd.DatetimeIndex

    scaler: object = None
    X_train: np.ndarray = field(default=None)
    X_val: np.ndarray = field(default=None)
    X_test: np.ndarray = field(default=None)

    @property
    def class_thresholds(self) -> tuple[float, float]:
        return demand_class_thresholds(self.y_train)

    def summary(self) -> dict:
        return {
            "n_train": int(len(self.y_train)),
            "n_val": int(len(self.y_val)),
            "n_test": int(len(self.y_test)),
            "train_period": [
                str(self.index_train[0]),
                str(self.index_train[-1]),
            ],
            "val_period": [
                str(self.index_val[0]),
                str(self.index_val[-1]),
            ],
            "test_period": [
                str(self.index_test[0]),
                str(self.index_test[-1]),
            ],
        }


def prepare_demand_split(
    df: pd.DataFrame | None = None,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
    scale: bool = True,
) -> DemandSplit:
    """Build chronological data partitions and fit scaler on training data only."""
    from sklearn.preprocessing import StandardScaler

    if df is None:
        df = load_and_clean()

    frame, features = build_demand_frame(df)

    validation_start, test_start = split_boundaries(
        frame.index,
        fractions,
    )

    train, validation, test = split_by_time(
        frame,
        validation_start,
        test_start,
    )

    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise ValueError("Train, validation, or test split is empty.")

    split = DemandSplit(
        feature_names=features,
        val_start=validation_start,
        test_start=test_start,
        X_train_raw=train[features].to_numpy(dtype=float),
        X_val_raw=validation[features].to_numpy(dtype=float),
        X_test_raw=test[features].to_numpy(dtype=float),
        y_train=train[TARGET_DEMAND].to_numpy(dtype=float),
        y_val=validation[TARGET_DEMAND].to_numpy(dtype=float),
        y_test=test[TARGET_DEMAND].to_numpy(dtype=float),
        index_train=train.index,
        index_val=validation.index,
        index_test=test.index,
    )

    if scale:
        scaler = StandardScaler()
        scaler.fit(split.X_train_raw)

        split.scaler = scaler
        split.X_train = scaler.transform(split.X_train_raw)
        split.X_val = scaler.transform(split.X_val_raw)
        split.X_test = scaler.transform(split.X_test_raw)
    else:
        split.X_train = split.X_train_raw
        split.X_val = split.X_val_raw
        split.X_test = split.X_test_raw

    return split


# ==========================================================================
# Approved model registry — only requested algorithms
# ==========================================================================
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def build_model(
    name: str,
    quick: bool = True,
    seed: int = RANDOM_SEED,
):
    """Create one approved individual regression model."""
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor

    if name == "Linear Regression":
        return LinearRegression()

    if name == "KNN Regressor":
        return KNeighborsRegressor(
            n_neighbors=8 if quick else 12,
            weights="distance",
            p=2,
            n_jobs=-1,
        )

    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=150 if quick else 500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=200 if quick else 600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            random_state=seed,
        )

    if name == "XGBoost":
        if not HAS_XGBOOST:
            raise RuntimeError(
                "XGBoost is not installed. Run: pip install xgboost"
            )

        return XGBRegressor(
            n_estimators=300 if quick else 900,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )

    raise KeyError(
        f"Unknown model '{name}'. "
        "Only KNN, Random Forest, Gradient Boosting, XGBoost, "
        "and Linear Regression are allowed."
    )


# Only these models appear in training and testing.
REGRESSION_MODELS = [
    "KNN Regressor",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "Linear Regression",
]

# train.py uses the same list for choosing the production model.
PRODUCTION_CANDIDATES = list(REGRESSION_MODELS)

# Do not display irrelevant algorithms in benchmark output.
NOT_APPLICABLE_MODELS = {}


# ==========================================================================
# Metrics and derived demand classes
# ==========================================================================
def tolerance_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Percentage of predictions within ±tolerance of actual demand."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    valid = np.abs(actual) > 1e-9

    if not np.any(valid):
        return float("nan")

    correct = (
        np.abs(predicted[valid] - actual[valid])
        <= tolerance * np.abs(actual[valid])
    )

    return float(np.mean(correct) * 100)


def demand_class_thresholds(y_train: np.ndarray) -> tuple[float, float]:
    """Create Low/Medium/High class thresholds from training targets only."""
    values = np.asarray(y_train, dtype=float)

    low_medium = float(np.quantile(values, 1 / 3))
    medium_high = float(np.quantile(values, 2 / 3))

    return low_medium, medium_high


def to_demand_classes(
    values: np.ndarray,
    thresholds: tuple[float, float],
) -> np.ndarray:
    """Convert continuous demand into Low=0, Medium=1, High=2."""
    low_medium, medium_high = thresholds
    values = np.asarray(values, dtype=float)

    return np.where(
        values < low_medium,
        0,
        np.where(values < medium_high, 1, 2),
    )


def derived_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: tuple[float, float],
) -> dict:
    """Derived classification metrics for Low/Medium/High demand bands."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    actual_classes = to_demand_classes(y_true, thresholds)
    predicted_classes = to_demand_classes(y_pred, thresholds)

    return {
        "derived_precision": float(
            precision_score(
                actual_classes,
                predicted_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "derived_recall": float(
            recall_score(
                actual_classes,
                predicted_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "derived_f1": float(
            f1_score(
                actual_classes,
                predicted_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "derived_class_accuracy": float(
            accuracy_score(actual_classes, predicted_classes)
        ),
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Return native regression metrics."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "tolerance_accuracy": tolerance_accuracy(
            actual,
            predicted,
            tolerance,
        ),
    }


def rank_key(row: dict):
    """Sort by accuracy descending, RMSE ascending, then R² descending."""
    accuracy = row.get("tolerance_accuracy")
    rmse = row.get("rmse")
    r2 = row.get("r2")

    if accuracy is None or not np.isfinite(accuracy):
        accuracy = -np.inf

    if rmse is None or not np.isfinite(rmse):
        rmse = np.inf

    if r2 is None or not np.isfinite(r2):
        r2 = -np.inf

    return (-accuracy, rmse, -r2)


# ==========================================================================
# Utility helpers
# ==========================================================================
class Timer:
    """Simple timing context manager."""

    def __enter__(self):
        self.started = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.seconds = time.perf_counter() - self.started
        return False


def utc_timestamp() -> str:
    """Return a UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def environment_info() -> dict:
    """Return basic runtime metadata."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def read_json_safe(path: Path | str):
    """Read JSON safely and return None for missing or invalid files."""
    path = Path(path)

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ==========================================================================
# Artifact staging and loading
# ==========================================================================
def staging_dir() -> Path:
    """Create a temporary staging folder inside models."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    path = MODEL_DIR / f".staging_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)

    return path


def promote_staged(staged: Path, filenames: list[str]) -> None:
    """Move successfully trained artifacts into the real models folder."""
    staged = Path(staged)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for filename in filenames:
            source = staged / filename

            if not source.exists():
                raise FileNotFoundError(
                    f"Expected staged artifact is missing: {source}"
                )

            destination = MODEL_DIR / filename
            shutil.move(str(source), str(destination))
    finally:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)


def load_production_bundle() -> dict:
    """Load all artifacts required by grid_sim.py."""
    required = PRODUCTION_FILES + ["model_metadata.json"]

    missing = [
        filename
        for filename in required
        if not (MODEL_DIR / filename).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Production model artifacts are missing:\n"
            + "\n".join(f"  - {filename}" for filename in missing)
            + "\nRun `python train.py` from the project folder first."
        )

    with open(MODEL_DIR / "features.json", "r", encoding="utf-8") as file:
        features = json.load(file)

    with open(MODEL_DIR / "last_weather.json", "r", encoding="utf-8") as file:
        last_weather = json.load(file)

    with open(MODEL_DIR / "model_metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return {
        "demand_model": joblib.load(MODEL_DIR / "demand_model.pkl"),
        "solar_model": joblib.load(MODEL_DIR / "solar_model.pkl"),
        "wind_model": joblib.load(MODEL_DIR / "wind_model.pkl"),
        "scaler_demand": joblib.load(MODEL_DIR / "scaler_demand.pkl"),
        "scaler_solar": joblib.load(MODEL_DIR / "scaler_solar.pkl"),
        "scaler_wind": joblib.load(MODEL_DIR / "scaler_wind.pkl"),
        "features": features,
        "last_weather": last_weather,
        "metadata": metadata,
    }