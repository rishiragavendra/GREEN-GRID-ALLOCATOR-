"""Export the trained solar/wind models + weather table to plain JSON.

The web app (local and Vercel) loads only these JSON files, so it needs no
xgboost / scikit-learn / pandas at runtime.  Run automatically by run.bat/run.sh
after training:   python export_runtime.py
"""
import json
import warnings
from pathlib import Path

import joblib

import ml_pipeline as mlp

warnings.filterwarnings("ignore")
OUT = mlp.ROOT / "runtime"


def flatten_tree(root):
    """Nested XGBoost JSON dump -> flat arrays (index = node id)."""
    nodes = {}

    def walk(n):
        nodes[n["nodeid"]] = n
        for c in n.get("children", []):
            walk(c)

    walk(root)
    size = max(nodes) + 1
    feat, thr, yes, no, miss, leaf = [-1] * size, [0.0] * size, [-1] * size, [-1] * size, [-1] * size, [0.0] * size
    for i, n in nodes.items():
        if "leaf" in n:
            leaf[i] = n["leaf"]
        else:
            feat[i] = int(n["split"][1:])
            thr[i] = n["split_condition"]
            yes[i], no[i], miss[i] = n["yes"], n["no"], n["missing"]
    return {"f": feat, "t": thr, "y": yes, "n": no, "m": miss, "l": leaf}


def scaler(s):
    return {"mean": [float(x) for x in s.mean_], "scale": [float(x) for x in s.scale_]}


def main():
    OUT.mkdir(exist_ok=True)
    b = mlp.load_production_bundle()
    solar = b["solar_model"]
    wind = b["wind_model"]

    meta = b["metadata"]
    model = {
        "features": {"solar": b["features"]["solar"], "wind": b["features"]["wind"]},
        "scaler_solar": scaler(b["scaler_solar"]),
        "scaler_wind": scaler(b["scaler_wind"]),
        "model_names": {k: meta.get(k, {}).get("production_model", "unknown") for k in ("demand", "solar", "wind")},
    }

    # Solar: XGBoost or any sklearn linear model
    if hasattr(solar, "get_booster"):
        booster = solar.get_booster()
        cfg = json.loads(booster.save_config())
        base = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
        model["solar"] = {"kind": "xgb", "base": base,
                          "trees": [flatten_tree(json.loads(t)) for t in booster.get_dump(dump_format="json")]}
    else:
        model["solar"] = {"kind": "linear", "coef": [float(x) for x in solar.coef_], "intercept": float(solar.intercept_)}

    if hasattr(wind, "coef_"):
        model["wind"] = {"kind": "linear", "coef": [float(x) for x in wind.coef_], "intercept": float(wind.intercept_)}
    else:
        raise SystemExit("Wind model is not linear - extend export_runtime.py for this model type.")

    (OUT / "model.json").write_text(json.dumps(model, separators=(",", ":")))

    df = mlp.load_and_clean(mlp.DATA_PATH)
    cols = mlp.WEATHER_COLS
    weather = {"cols": cols,
               "times": [t.strftime("%Y-%m-%d %H:%M") for t in df.index],
               "rows": [[float(v) for v in r] for r in df[cols].values]}
    (OUT / "weather.json").write_text(json.dumps(weather, separators=(",", ":")))

    perf = mlp.read_json_safe(mlp.BENCHMARK_JSON)
    if perf:
        (OUT / "benchmark_results.json").write_text(json.dumps(perf, separators=(",", ":")))
    (OUT / "model_metadata.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("Runtime export written to", OUT)


if __name__ == "__main__":
    main()
