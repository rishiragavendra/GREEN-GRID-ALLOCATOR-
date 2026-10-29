"""Train and persist the production models for the Green Grid Allocator.

    python train.py                 # default (fast) hyper-parameters
    python train.py --full          # larger forests / boosting rounds
    python train.py --tolerance 0.05

What it does
------------
1. Loads and cleans the Tetuan City hourly dataset.
2. Builds the shared feature set from `ml_pipeline` (weather + cyclical time +
   lag 24/168 + rolling means 24/168).
3. Splits chronologically into train / validation / test (70/15/15) - the data
   are never shuffled.
4. Trains several traditional ML regressors on TRAIN and scores them on
   VALIDATION using the project ranking rule
   (tolerance accuracy -> RMSE -> R-squared).
5. Refits the winner on TRAIN+VALIDATION and reports untouched TEST metrics.
6. Trains the solar and wind capacity-factor models the simulator needs.
7. Writes everything to `models/` only after all training succeeded.

No deep learning is used anywhere in this project - see the report in `docs/`.
"""

from __future__ import annotations

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import argparse
import json
import joblib
import numpy as np

import ml_pipeline as mlp


# --------------------------------------------------------------------------
def _fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def select_demand_model(split: mlp.DemandSplit, quick: bool, tolerance: float):
    """Train every production candidate on TRAIN, rank them on VALIDATION."""
    rows = []
    print(f"\n--- Selecting the demand model ({len(mlp.PRODUCTION_CANDIDATES)} candidates) ---")
    print(f"{'model':<26} {'tol.acc%':>9} {'RMSE':>10} {'R2':>8} {'fit s':>7}")
    print("-" * 64)

    for name in mlp.PRODUCTION_CANDIDATES:
        try:
            model = mlp.build_model(name, quick=quick)
            with mlp.Timer() as t:
                model.fit(split.X_train, split.y_train)
            pred = model.predict(split.X_val)
            m = mlp.regression_metrics(split.y_val, pred, tolerance)
            m.update(model=name, fit_seconds=round(t.seconds, 2))
            rows.append(m)
            print(f"{name:<26} {m['tolerance_accuracy']:>9.2f} {m['rmse']:>10.1f} "
                  f"{m['r2']:>8.4f} {t.seconds:>7.1f}")
        except Exception as exc:  # a candidate failing must not kill training
            print(f"{name:<26} {'FAILED':>9}  ({type(exc).__name__}: {exc})")

    if not rows:
        raise RuntimeError("No demand candidate trained successfully.")
    rows.sort(key=mlp.rank_key)
    return rows[0]["model"], rows


def train_renewable(name_list, X_tr, y_tr, X_va, y_va, quick, label):
    """Pick the best small regressor for a capacity-factor target."""
    best = best_key = best_metrics = best_name = None
    for name in name_list:
        try:
            model = mlp.build_model(name, quick=quick)
            model.fit(X_tr, y_tr)
            m = mlp.regression_metrics(y_va, model.predict(X_va), 0.10)
            key = (m["rmse"], -m["r2"])
            if best_key is None or key < best_key:
                best, best_key, best_metrics, best_name = model, key, m, name
        except Exception as exc:
            print(f"  [{label}] {name} failed: {type(exc).__name__}: {exc}")
    if best is None:
        raise RuntimeError(f"No {label} candidate trained successfully.")
    print(f"  [{label}] selected {best_name}: RMSE {best_metrics['rmse']:.4f}  "
          f"R2 {best_metrics['r2']:.4f}")
    return best, best_name, best_metrics


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train Green Grid Allocator models")
    ap.add_argument("--full", action="store_true",
                    help="use heavier hyper-parameters (slower, usually better)")
    ap.add_argument("--tolerance", type=float, default=mlp.DEFAULT_TOLERANCE,
                    help="relative band for accuracy-within-tolerance "
                         "(default 0.10 = +/-10%%)")
    args = ap.parse_args()
    quick = not args.full
    tol = args.tolerance

    np.random.seed(mlp.RANDOM_SEED)

    print("=" * 70)
    print("  GREEN GRID ALLOCATOR - production training")
    print(f"  mode={'full' if args.full else 'quick'}  tolerance=+/-{tol*100:.0f}%  "
          f"seed={mlp.RANDOM_SEED}")
    print("=" * 70)

    print("\nLoading data...")
    df = mlp.load_and_clean()
    print(f"  hourly rows: {len(df)}  ({df.index[0]} -> {df.index[-1]})")

    split = mlp.prepare_demand_split(df)
    s = split.summary()
    print(f"  features: {len(split.feature_names)}")
    print(f"  train {s['n_train']}  |  val {s['n_val']}  |  test {s['n_test']}")
    print(f"  test period held out: {s['test_period'][0]} -> {s['test_period'][1]}")

    # ---------------- demand ----------------
    best_name, candidate_rows = select_demand_model(split, quick, tol)
    print(f"\n  -> selected '{best_name}' by {mlp.RANKING_RULE_TEXT}")

    # Refit the winner on TRAIN + VALIDATION for the production artifact.
    from sklearn.preprocessing import StandardScaler
    X_fit_raw = np.vstack([split.X_train_raw, split.X_val_raw])
    y_fit = np.concatenate([split.y_train, split.y_val])
    scaler_demand = StandardScaler().fit(X_fit_raw)
    demand_model = mlp.build_model(best_name, quick=quick)
    print(f"  refitting '{best_name}' on train+validation ({len(y_fit)} rows)...")
    demand_model.fit(scaler_demand.transform(X_fit_raw), y_fit)

    test_pred = demand_model.predict(scaler_demand.transform(split.X_test_raw))
    test_metrics = mlp.regression_metrics(split.y_test, test_pred, tol)
    thresholds = mlp.demand_class_thresholds(y_fit)
    test_metrics.update(mlp.derived_class_metrics(split.y_test, test_pred, thresholds))
    print(f"  held-out TEST: tol.acc {test_metrics['tolerance_accuracy']:.2f}%  "
          f"MAE {test_metrics['mae']:.1f}  RMSE {test_metrics['rmse']:.1f}  "
          f"R2 {test_metrics['r2']:.4f}")

    # ---------------- solar / wind ----------------
    print("\n--- Renewable capacity-factor models ---")
    # Solar and wind use only fast approved models.
    # Demand forecasting still compares all five algorithms.
    renewable_candidates = [
    "XGBoost",
    "Linear Regression",
]
    solar_frame, solar_feats = mlp.build_solar_frame(df)
    s_tr, s_va, s_te = mlp.split_by_time(solar_frame, split.val_start, split.test_start)
    scaler_solar = StandardScaler().fit(s_tr[solar_feats].to_numpy(dtype=float))
    solar_model, solar_name, solar_val = train_renewable(
        renewable_candidates,
        scaler_solar.transform(s_tr[solar_feats].to_numpy(dtype=float)),
        s_tr["solar_cf"].to_numpy(dtype=float),
        scaler_solar.transform(s_va[solar_feats].to_numpy(dtype=float)),
        s_va["solar_cf"].to_numpy(dtype=float), quick, "solar")
    solar_test = mlp.regression_metrics(
        s_te["solar_cf"].to_numpy(dtype=float),
        solar_model.predict(scaler_solar.transform(
            s_te[solar_feats].to_numpy(dtype=float))), tol)

    wind_frame, wind_feats = mlp.build_wind_frame(df)
    w_tr, w_va, w_te = mlp.split_by_time(wind_frame, split.val_start, split.test_start)
    scaler_wind = StandardScaler().fit(w_tr[wind_feats].to_numpy(dtype=float))
    wind_model, wind_name, wind_val = train_renewable(
        renewable_candidates,
        scaler_wind.transform(w_tr[wind_feats].to_numpy(dtype=float)),
        w_tr["wind_cf"].to_numpy(dtype=float),
        scaler_wind.transform(w_va[wind_feats].to_numpy(dtype=float)),
        w_va["wind_cf"].to_numpy(dtype=float), quick, "wind")
    wind_test = mlp.regression_metrics(
        w_te["wind_cf"].to_numpy(dtype=float),
        wind_model.predict(scaler_wind.transform(
            w_te[wind_feats].to_numpy(dtype=float))), tol)

    # ---------------- persist (staged, then promoted) ----------------
    print("\n--- Saving artifacts ---")
    staged = mlp.staging_dir()

    joblib.dump(demand_model, staged / "demand_model.pkl")
    joblib.dump(solar_model, staged / "solar_model.pkl")
    joblib.dump(wind_model, staged / "wind_model.pkl")
    joblib.dump(scaler_demand, staged / "scaler_demand.pkl")
    joblib.dump(scaler_solar, staged / "scaler_solar.pkl")
    joblib.dump(scaler_wind, staged / "scaler_wind.pkl")

    with open(staged / "features.json", "w") as f:
        json.dump({"demand": split.feature_names,
                   "solar": solar_feats,
                   "wind": wind_feats}, f, indent=2)

    last_row = df.iloc[-1][mlp.WEATHER_COLS].to_dict()
    with open(staged / "last_weather.json", "w") as f:
        json.dump({k: float(v) for k, v in last_row.items()}, f, indent=2)

    metadata = {
        "artifact_version": mlp.ARTIFACT_VERSION,
        "trained_at_utc": mlp.utc_timestamp(),
        "approach": "traditional machine learning only "
                    "(no deep learning, no image features)",
        "random_seed": mlp.RANDOM_SEED,
        "training_mode": "full" if args.full else "quick",
        "tolerance": tol,
        "dataset": {
            "file": mlp.DATA_PATH.name,
            "hourly_rows": int(len(df)),
            "range": [str(df.index[0]), str(df.index[-1])],
            "resampling": "1h mean of 10-minute readings, forward/back filled",
        },
        "split": {"fractions": list(mlp.SPLIT_FRACTIONS), "shuffled": False, **s},
        "scaler": {"type": "StandardScaler",
                   "demand_fitted_on": "train+validation",
                   "solar_wind_fitted_on": "train"},
        "demand": {
            "target": mlp.TARGET_DEMAND,
            "unit": "kW (converted to MW by the simulator)",
            "production_model": best_name,
            "selection_rule": mlp.RANKING_RULE_TEXT,
            "n_features": len(split.feature_names),
            "features": split.feature_names,
            "validation_candidates": candidate_rows,
            "test_metrics": test_metrics,
            "class_thresholds": {"low_medium": thresholds[0],
                                 "medium_high": thresholds[1],
                                 "source": "tertiles of the training(+val) target"},
        },
        "solar": {"target": "solar_cf (synthetic capacity factor 0-1)",
                  "production_model": solar_name, "features": solar_feats,
                  "validation_metrics": solar_val, "test_metrics": solar_test},
        "wind": {"target": "wind_cf (synthetic capacity factor 0-1)",
                 "production_model": wind_name, "features": wind_feats,
                 "validation_metrics": wind_val, "test_metrics": wind_test},
        "environment": mlp.environment_info(),
    }
    with open(staged / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    mlp.promote_staged(staged, mlp.PRODUCTION_FILES + ["model_metadata.json"])

    print(f"  written to {mlp.MODEL_DIR}")
    for f in mlp.PRODUCTION_FILES + ["model_metadata.json"]:
        print(f"    - {f}")

    print("\n" + "=" * 70)
    print(f"  Production demand model : {best_name}")
    print(f"  Solar model             : {solar_name}")
    print(f"  Wind model              : {wind_name}")
    print(f"  Demand test R2          : {_fmt(test_metrics['r2'])}")
    print(f"  Demand test RMSE        : {test_metrics['rmse']:.1f} kW")
    print(f"  Demand tol.accuracy     : {test_metrics['tolerance_accuracy']:.2f}% "
          f"(+/-{tol*100:.0f}%)")
    print("=" * 70)
    print("\nNext:  python test.py --quick     (full benchmark + UI metrics)")
    print("       python app.py               (web UI at http://127.0.0.1:8000)")


if __name__ == "__main__":
    main()
