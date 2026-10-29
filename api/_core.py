"""Stateless, dependency-free simulator core (pure Python).

Used by the FastAPI app locally and on Vercel.  All simulator state lives in a
plain dict that the browser holds and sends with each request:

    {"sources": [...], "demand_mw": 100, "index": 8735, "weather": {...}, "history": [...]}

Models come from runtime/*.json (written by export_runtime.py), so no xgboost,
scikit-learn or pandas is needed to serve the app.
"""
import copy
import json
import math
import struct
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"

WEATHER_COLS = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows"]
HISTORY_MAX = 500
BATTERY_EFFICIENCY = 0.9

SOURCE_TYPES = [
    {"id": 1, "name": "Solar", "icon": "[SOL]", "emission": 0.0, "cost": 0, "renewable": True},
    {"id": 2, "name": "Wind", "icon": "[WND]", "emission": 0.0, "cost": 0, "renewable": True},
    {"id": 3, "name": "Hydro", "icon": "[HYD]", "emission": 0.0, "cost": 5, "renewable": True},
    {"id": 4, "name": "Battery", "icon": "[BAT]", "emission": 0.0, "cost": 0, "renewable": True, "storage": True},
    {"id": 5, "name": "Gas CCGT", "icon": "[GAS]", "emission": 0.4, "cost": 50, "renewable": False},
    {"id": 6, "name": "Coal", "icon": "[COA]", "emission": 0.9, "cost": 40, "renewable": False},
    {"id": 7, "name": "Diesel", "icon": "[DIE]", "emission": 0.7, "cost": 80, "renewable": False},
]
DEFAULT_SOURCES = [
    {"type_id": 1, "capacity_mw": 50},
    {"type_id": 2, "capacity_mw": 30},
    {"type_id": 3, "capacity_mw": 20},
    {"type_id": 4, "capacity_mw": 10, "soc": 0.5},
    {"type_id": 5, "capacity_mw": 100},
    {"type_id": 6, "capacity_mw": 80},
    {"type_id": 7, "capacity_mw": 20},
]

_cache = {}


def _load(name):
    if name not in _cache:
        path = RUNTIME / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `python export_runtime.py` (run.bat does this automatically).")
        _cache[name] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[name]


def model():
    return _load("model.json")


def weather_table():
    return _load("weather.json")


def read_optional(name):
    try:
        return _load(name)
    except (FileNotFoundError, ValueError):
        return None


# ------------------------------------------------------------------ models
def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _time_feats(ts):
    """ts is 'YYYY-MM-DD HH:MM'."""
    hour, month = int(ts[11:13]), int(ts[5:7])
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24),
        "month_sin": math.sin(2 * math.pi * month / 12), "month_cos": math.cos(2 * math.pi * month / 12),
    }


def _scaled(names, scaler, feats):
    return [(feats.get(n, 0.0) - m) / s for n, m, s in zip(names, scaler["mean"], scaler["scale"])]


def _predict(spec, x):
    if spec["kind"] == "linear":
        return spec["intercept"] + sum(c * v for c, v in zip(spec["coef"], x))
    xf = [_f32(v) for v in x]
    total = spec["base"]
    for t in spec["trees"]:
        n = 0
        while t["f"][n] >= 0:
            v = xf[t["f"][n]]
            n = t["y"][n] if v < _f32(t["t"][n]) else t["n"][n]
        total += t["l"][n]
    return total


def solar_cf(weather, ts):
    m = model()
    feats = {**weather, **_time_feats(ts)}
    return min(1.0, max(0.0, _predict(m["solar"], _scaled(m["features"]["solar"], m["scaler_solar"], feats))))


def wind_cf(weather, ts):
    m = model()
    feats = {**weather, **_time_feats(ts)}
    return min(1.0, max(0.0, _predict(m["wind"], _scaled(m["features"]["wind"], m["scaler_wind"], feats))))


# ------------------------------------------------------------------ state
def make_source(type_id, capacity, new_id, soc=None):
    st = SOURCE_TYPES[type_id - 1]
    storage = st.get("storage", False)
    return {
        "id": new_id, "type_id": type_id, "name": st["name"], "icon": st["icon"],
        "capacity": capacity, "available": 0.0, "used": 0.0, "emission": st["emission"],
        "cost": st["cost"], "renewable": st["renewable"], "storage": storage,
        "soc": soc if soc is not None else (0.5 if storage else 0.0),
        "max_charge": capacity, "max_discharge": capacity,
    }


def weather_at(idx):
    w = weather_table()
    return dict(zip(w["cols"], w["rows"][idx]))


def default_state():
    n = len(weather_table()["times"])
    idx = n - 1
    return {
        "sources": [make_source(s["type_id"], s["capacity_mw"], i, s.get("soc"))
                    for i, s in enumerate(DEFAULT_SOURCES)],
        "demand_mw": 100.0, "index": idx, "weather": weather_at(idx), "history": [],
    }


def clean_state(raw):
    """Validate/coerce a client-supplied state; fall back to defaults for gaps."""
    base = default_state()
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        return base
    n = len(weather_table()["times"])
    st = {
        "sources": raw["sources"],
        "demand_mw": max(0.0, float(raw.get("demand_mw", 100.0))),
        "index": int(raw.get("index", raw.get("current_idx", base["index"]))) % n,
        "history": raw.get("history") if isinstance(raw.get("history"), list) else [],
    }
    w = raw.get("weather") if isinstance(raw.get("weather"), dict) else {}
    st["weather"] = {c: float(w[c]) if c in w and w[c] is not None else base["weather"][c] for c in WEATHER_COLS}
    for s in st["sources"]:
        for k in ("capacity", "soc", "max_charge", "max_discharge", "emission", "cost"):
            s[k] = float(s.get(k, 0.0))
        s["storage"], s["renewable"] = bool(s.get("storage")), bool(s.get("renewable"))
    return st


# ------------------------------------------------------------------ dispatch
def allocate(state):
    """Merit-order dispatch; mutates state['sources'] (used/available/soc)."""
    ts = weather_table()["times"][state["index"]]
    scf, wcf = solar_cf(state["weather"], ts), wind_cf(state["weather"], ts)
    sources, demand = state["sources"], state["demand_mw"]

    for s in sources:
        t = s["type_id"]
        if t == 1:
            s["available"] = s["capacity"] * scf
        elif t == 2:
            s["available"] = s["capacity"] * wcf
        elif t == 3:
            s["available"] = s["capacity"] * 0.9
        elif t == 4:
            s["available"] = s["soc"] * s["capacity"]
        else:
            s["available"] = s["capacity"]
        s["used"] = 0.0

    total_renewable = total_fossil = co2 = charged = discharged = curtailed = 0.0

    for s in sources:  # phase 1: renewables at full output
        if s["renewable"] and not s["storage"]:
            s["used"] = s["available"]
            total_renewable += s["available"]

    surplus = max(0.0, total_renewable - demand)
    deficit = max(0.0, demand - total_renewable)

    for s in sources:  # phase 2: battery
        if not s["storage"]:
            continue
        if surplus > 0:
            headroom = (1.0 - s["soc"]) * s["capacity"]
            charge = min(surplus, headroom, s.get("max_charge", s["capacity"]))
            if charge > 0:
                s["used"] = -charge
                s["soc"] = min(1.0, s["soc"] + (charge * BATTERY_EFFICIENCY) / s["capacity"])
                charged += charge
                surplus -= charge
            curtailed = surplus
        elif deficit > 0:
            d = min(deficit, s["soc"] * s["capacity"], s.get("max_discharge", s["capacity"]))
            if d > 0:
                s["used"] = d
                s["soc"] = max(0.0, s["soc"] - d / s["capacity"])
                discharged += d
                deficit -= d
                total_renewable += d

    remaining = deficit
    for s in sorted((s for s in sources if not s["renewable"]), key=lambda x: x["cost"]):  # phase 3
        used = min(s["available"], remaining)
        s["used"] = used
        total_fossil += used
        co2 += used * s["emission"]
        remaining -= used
        if remaining <= 0:
            break

    return {
        "demand": demand,
        "renewable_used": min(total_renewable, demand),
        "renewable_generated": total_renewable,
        "fossil_used": total_fossil,
        "battery_charged": charged,
        "battery_discharged": discharged,
        "curtailed": curtailed,
        "unmet": max(0, remaining),
        "co2": co2,
        "renewable_pct": (min(total_renewable, demand) / demand * 100) if demand > 0 else 0,
    }


def snapshot(state, commit=False):
    """Run dispatch and return the full JSON-safe snapshot (which is also the next state).

    commit=False dispatches on a throwaway copy so the battery SoC is not moved;
    commit=True (time change) keeps the post-dispatch sources.
    """
    if commit:
        res = allocate(state)
        out_sources = state["sources"]
    else:
        work = {**state, "sources": copy.deepcopy(state["sources"])}
        res = allocate(work)
        out_sources = work["sources"]
        for s_out, s_orig in zip(out_sources, state["sources"]):
            s_out["soc"] = s_orig["soc"]  # a poll/refresh must not move the battery

    times = weather_table()["times"]
    history = list(state.get("history", []))
    if not history or history[-1].get("idx") != state["index"]:
        batt = next((s for s in out_sources if s.get("storage")), None)
        history.append({
            "idx": state["index"],
            "ts": _epoch(times[state["index"]]),
            "t": times[state["index"]][5:],
            "renewable_pct": round(res["renewable_pct"], 1),
            "renewable_mw": round(res["renewable_used"], 1),
            "fossil_mw": round(res["fossil_used"], 1),
            "demand": round(res["demand"], 1),
            "curtailed": round(res["curtailed"], 1),
            "co2": round(res["co2"], 2),
            "battery_soc": round(batt["soc"] * 100, 1) if batt else 0.0,
        })
        history = history[-HISTORY_MAX:]

    ts = times[state["index"]]
    return {
        "time": ts, "weekday": _weekday(ts), "index": state["index"], "total_hours": len(times),
        "demand_mw": state["demand_mw"], "weather": state["weather"],
        "sources": out_sources, "dispatch": res, "history": history,
    }


def _epoch(ts):
    import calendar
    import time as _t
    return float(calendar.timegm(_t.strptime(ts, "%Y-%m-%d %H:%M")))


def _weekday(ts):
    import datetime
    return datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M").strftime("%A")


def index_for_datetime(text):
    """Nearest-at-or-after index for an ISO-like datetime string."""
    norm = str(text).replace("T", " ")[:16]
    times = weather_table()["times"]
    if len(norm) == 13:
        norm += ":00"
    return min(bisect_left(times, norm), len(times) - 1)
