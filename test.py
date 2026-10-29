"""Benchmark only approved machine-learning models for demand forecasting.

Run:
    python test.py
    python test.py --quick
    python test.py --full
    python test.py --tolerance 0.10
"""

from __future__ import annotations

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import argparse
import csv
import json
import random
import numpy as np

import ml_pipeline as mlp

REG = "regressor"
ENS = "ensemble"


def blank_row(name: str, model_type: str, status: str = "") -> dict:
    return {
        "model": name,
        "type": model_type,
        "mae": None,
        "rmse": None,
        "r2": None,
        "tolerance_accuracy": None,
        "derived_precision": None,
        "derived_recall": None,
        "derived_f1": None,
        "derived_class_accuracy": None,
        "fit_seconds": None,
        "predict_seconds": None,
        "rank": None,
        "rankable": False,
        "status": status,
    }


def evaluate_model(name, model_type, y_true, y_pred, thresholds, tolerance,
                   fit_seconds=None, predict_seconds=None) -> dict:
    row = blank_row(name, model_type, "ok")
    row.update(mlp.regression_metrics(y_true, y_pred, tolerance))
    row.update(mlp.derived_class_metrics(y_true, y_pred, thresholds))
    row["fit_seconds"] = round(float(fit_seconds), 3) if fit_seconds is not None else None
    row["predict_seconds"] = round(float(predict_seconds), 4) if predict_seconds is not None else None
    row["rankable"] = True
    return row


def train_individual_models(split, quick, tolerance, thresholds):
    rows = []
    predictions = {}

    print("\n[1/3] Training approved individual algorithms")
    print("-" * 85)

    for index, name in enumerate(mlp.REGRESSION_MODELS, start=1):
        print(f"({index}/{len(mlp.REGRESSION_MODELS)}) Training {name}...")

        try:
            model = mlp.build_model(name, quick=quick)

            with mlp.Timer() as fit_timer:
                model.fit(split.X_train, split.y_train)

            validation_predictions = model.predict(split.X_val)

            with mlp.Timer() as predict_timer:
                test_predictions = model.predict(split.X_test)

            row = evaluate_model(
                name=name,
                model_type=REG,
                y_true=split.y_test,
                y_pred=test_predictions,
                thresholds=thresholds,
                tolerance=tolerance,
                fit_seconds=fit_timer.seconds,
                predict_seconds=predict_timer.seconds,
            )

            rows.append(row)
            predictions[name] = {
                "validation": np.asarray(validation_predictions, dtype=float),
                "test": np.asarray(test_predictions, dtype=float),
            }

            print(
                f"  Accuracy: {row['tolerance_accuracy']:.2f}% | "
                f"RMSE: {row['rmse']:.2f} | R²: {row['r2']:.4f}"
            )

        except Exception as exc:
            rows.append(blank_row(name, REG, f"failed: {type(exc).__name__}: {exc}"))
            print(f"  Failed: {exc}")

    return rows, predictions


def fit_ensemble_weights(validation_matrix, y_validation):
    """Find non-negative weights from validation data only."""
    count = validation_matrix.shape[1]
    equal_weights = np.full(count, 1.0 / count)

    def objective(weights):
        prediction = validation_matrix @ weights
        return float(np.sqrt(np.mean((prediction - y_validation) ** 2)))

    try:
        from scipy.optimize import minimize

        result = minimize(
            objective,
            equal_weights,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * count,
            constraints=[{
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - 1.0),
            }],
        )

        if result.success and np.sum(result.x) > 0:
            weights = np.clip(result.x, 0, None)
            return weights / weights.sum()

    except Exception:
        pass

    return equal_weights


def build_ensembles(rows, predictions, split, tolerance, thresholds):
    print("\n[2/3] Building approved ensemble combinations")
    print("-" * 85)

    successful = [
        row for row in rows
        if row["type"] == REG and row["status"] == "ok" and row["model"] in predictions
    ]
    successful.sort(key=mlp.rank_key)

    top_three = [row["model"] for row in successful[:3]]

    combinations = [
        ("Ensemble: Random Forest + XGBoost", ["Random Forest", "XGBoost"]),
        ("Ensemble: Gradient Boosting + XGBoost", ["Gradient Boosting", "XGBoost"]),
        ("Ensemble: Top 3 Models", top_three),
    ]

    ensemble_rows = []

    for label, members in combinations:
        missing = [member for member in members if member not in predictions]

        if len(members) < 2 or missing:
            ensemble_rows.append(
                blank_row(label, ENS, f"skipped: unavailable models: {missing}")
            )
            print(f"{label}: skipped")
            continue

        validation_matrix = np.column_stack(
            [predictions[member]["validation"] for member in members]
        )
        test_matrix = np.column_stack(
            [predictions[member]["test"] for member in members]
        )

        weights = fit_ensemble_weights(validation_matrix, split.y_val)
        ensemble_predictions = test_matrix @ weights

        row = evaluate_model(
            name=label,
            model_type=ENS,
            y_true=split.y_test,
            y_pred=ensemble_predictions,
            thresholds=thresholds,
            tolerance=tolerance,
        )

        row["members"] = members
        row["weights"] = [round(float(weight), 4) for weight in weights]
        row["status"] = (
            "Validation-weighted ensemble: "
            + ", ".join(
                f"{member}={weight:.3f}"
                for member, weight in zip(members, weights)
            )
        )

        ensemble_rows.append(row)

        print(
            f"{label}: Accuracy {row['tolerance_accuracy']:.2f}% | "
            f"RMSE {row['rmse']:.2f} | R² {row['r2']:.4f}"
        )

    return ensemble_rows


def print_results(rows, tolerance):
    print("\n" + "=" * 125)
    print("BENCHMARK RESULTS — APPROVED ALGORITHMS ONLY")
    print(f"Tolerance accuracy: prediction within ±{tolerance * 100:.0f}% of actual demand")
    print("Precision, recall and F1 are derived Low/Medium/High demand-class metrics.")
    print("=" * 125)

    print(
        f"{'Rank':<6} {'Model':<43} {'Type':<12} "
        f"{'Accuracy %':>11} {'RMSE':>12} {'R²':>10} "
        f"{'Precision':>11} {'Recall':>10} {'F1':>10}"
    )
    print("-" * 125)

    for row in rows:
        if row["status"] != "ok" and not row.get("rankable"):
            continue

        rank = row["rank"] if row["rank"] else "-"
        print(
            f"{str(rank):<6} "
            f"{row['model']:<43} "
            f"{row['type']:<12} "
            f"{row['tolerance_accuracy']:>10.2f}% "
            f"{row['rmse']:>12.2f} "
            f"{row['r2']:>10.4f} "
            f"{row['derived_precision']:>11.4f} "
            f"{row['derived_recall']:>10.4f} "
            f"{row['derived_f1']:>10.4f}"
        )

    print("=" * 125)


def best_reason(row, tolerance):
    return (
        f"{row['model']} was selected using the ranking rule: highest "
        f"accuracy within ±{tolerance * 100:.0f}%, then lowest RMSE, then "
        f"highest R². It achieved {row['tolerance_accuracy']:.2f}% accuracy, "
        f"RMSE {row['rmse']:.2f}, and R² {row['r2']:.4f}."
    )


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=mlp.DEFAULT_TOLERANCE,
        help="Relative tolerance, e.g. 0.10 means ±10%%",
    )
    parser.add_argument("--seed", type=int, default=mlp.RANDOM_SEED)
    args = parser.parse_args()

    quick = not args.full
    tolerance = args.tolerance

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 85)
    print("GREEN GRID ALLOCATOR — SELECTED ALGORITHM BENCHMARK")
    print("Algorithms: KNN, Random Forest, Gradient Boosting, XGBoost, Linear Regression")
    print("Plus validation-weighted ensemble combinations.")
    print("=" * 85)

    dataframe = mlp.load_and_clean()
    split = mlp.prepare_demand_split(dataframe)
    thresholds = split.class_thresholds
    summary = split.summary()

    print(f"\nDataset: {mlp.DATA_PATH.name}")
    print(f"Target: {mlp.TARGET_DEMAND}")
    print(f"Train / validation / test: {summary['n_train']} / {summary['n_val']} / {summary['n_test']}")

    individual_rows, predictions = train_individual_models(
        split, quick, tolerance, thresholds
    )

    ensemble_rows = build_ensembles(
        individual_rows, predictions, split, tolerance, thresholds
    )

    print("\n[3/3] Ranking and saving results")

    rankable_rows = [
        row for row in individual_rows + ensemble_rows
        if row.get("rankable")
    ]
    rankable_rows.sort(key=mlp.rank_key)

    for rank, row in enumerate(rankable_rows, start=1):
        row["rank"] = rank

    all_rows = rankable_rows + [
        row for row in individual_rows + ensemble_rows
        if not row.get("rankable")
    ]

    print_results(all_rows, tolerance)

    best_individual = next(
        (row for row in rankable_rows if row["type"] == REG),
        None,
    )
    best_ensemble = next(
        (row for row in rankable_rows if row["type"] == ENS),
        None,
    )

    if best_individual:
        print(f"\nBEST INDIVIDUAL MODEL: {best_individual['model']}")
        print(best_reason(best_individual, tolerance))

    if best_ensemble:
        print(f"\nBEST ENSEMBLE MODEL: {best_ensemble['model']}")
        print(best_reason(best_ensemble, tolerance))

    payload = {
        "schema_version": "2.0",
        "generated_at_utc": mlp.utc_timestamp(),
        "mode": "full" if args.full else "quick",
        "tolerance": tolerance,
        "tolerance_label": f"within ±{tolerance * 100:.0f}% of actual demand",
        "ranking_rule": (
            "Highest tolerance accuracy, then lowest RMSE, then highest R²."
        ),
        "dataset": {
            "file": mlp.DATA_PATH.name,
            "target": mlp.TARGET_DEMAND,
            "target_unit": "kW",
            "hourly_rows": int(len(dataframe)),
        },
        "split": {
            "fractions": list(mlp.SPLIT_FRACTIONS),
            "shuffled": False,
            **summary,
        },
        "demand_class_thresholds": {
            "low_medium": float(thresholds[0]),
            "medium_high": float(thresholds[1]),
            "source": "training-set tertiles only",
        },
        "derived_metrics_note": (
            "Precision, recall, F1 and classification accuracy are derived by "
            "converting continuous demand values into Low, Medium and High classes."
        ),
        "approved_algorithms": [
            "KNN Regressor",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "Linear Regression",
        ],
        "results": all_rows,
        "best_individual": best_individual,
        "best_individual_reason": (
            best_reason(best_individual, tolerance) if best_individual else None
        ),
        "best_ensemble": best_ensemble,
        "best_ensemble_reason": (
            best_reason(best_ensemble, tolerance) if best_ensemble else None
        ),
    }

    mlp.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(mlp.BENCHMARK_JSON, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    columns = [
        "rank", "model", "type", "mae", "rmse", "r2",
        "tolerance_accuracy", "derived_precision", "derived_recall",
        "derived_f1", "derived_class_accuracy", "fit_seconds",
        "predict_seconds", "status",
    ]

    with open(mlp.BENCHMARK_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved benchmark JSON: {mlp.BENCHMARK_JSON}")
    print(f"Saved benchmark CSV:  {mlp.BENCHMARK_CSV}")


if __name__ == "__main__":
    main()