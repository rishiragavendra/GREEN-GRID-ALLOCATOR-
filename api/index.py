"""Green Grid FastAPI API and dashboard for local use and Vercel."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import _core as core  # noqa: E402

ROOT = core.ROOT
PUBLIC = ROOT / "public"

app = FastAPI(title="Green Grid Allocator")
from starlette.middleware.base import BaseHTTPMiddleware

class VercelPathFixMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.scope["path"]

        if path in ("/api/index", "/api/index.py"):
            original_path = request.headers.get("x-vercel-original-path")

            if original_path:
                request.scope["path"] = original_path

        return await call_next(request)

app.add_middleware(VercelPathFixMiddleware)


def _state(body):
    return core.clean_state((body or {}).get("state"))


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/state")
def get_state():
    """Default grid at the latest hour (the browser keeps state after that)."""
    return core.snapshot(core.default_state())


@app.post("/api/state")
def refresh_state(body: dict = Body(default=None)):
    return core.snapshot(_state(body))


@app.post("/api/time")
def set_time(body: dict = Body(...)):
    """Advance/rewind by `hours` (default +1) or jump to an ISO `datetime`."""
    st = _state(body)
    n = len(core.weather_table()["times"])
    if body.get("datetime"):
        st["index"] = core.index_for_datetime(body["datetime"])
    else:
        st["index"] = (st["index"] + int(body.get("hours", 1))) % n
    st["weather"] = core.weather_at(st["index"])
    return core.snapshot(st, commit=True)


@app.post("/api/demand")
def set_demand(body: dict = Body(...)):
    st = _state(body)
    if body.get("set") is not None:
        st["demand_mw"] = max(0.0, float(body["set"]))
    elif body.get("delta"):
        st["demand_mw"] = max(0.0, st["demand_mw"] + float(body["delta"]))
    return core.snapshot(st)


@app.post("/api/weather")
def set_weather(body: dict = Body(...)):
    st = _state(body)
    for k, v in body.items():
        if k in core.WEATHER_COLS and v is not None and v != "":
            try:
                st["weather"][k] = float(v)
            except (TypeError, ValueError):
                continue
    return core.snapshot(st)


@app.post("/api/sources")
def add_source(body: dict = Body(...)):
    st = _state(body)
    try:
        tid, cap = int(body.get("type_id", 0)), float(body.get("capacity", 0))
    except (TypeError, ValueError):
        raise HTTPException(400, "type_id and capacity must be numbers")
    if not (1 <= tid <= len(core.SOURCE_TYPES)) or not (cap == cap and 0 < cap < float("inf")):
        raise HTTPException(400, "Invalid type_id (1-7) or capacity (>0)")
    new_id = max([s.get("id", 0) for s in st["sources"]], default=-1) + 1
    st["sources"].append(core.make_source(tid, cap, new_id))
    return core.snapshot(st)


@app.post("/api/sources/remove")
def remove_source(body: dict = Body(...)):
    st = _state(body)
    try:
        sid = int(body.get("id", -1))
    except (TypeError, ValueError):
        raise HTTPException(400, "id must be a number")
    keep = [s for s in st["sources"] if s.get("id") != sid]
    if len(keep) == len(st["sources"]):
        raise HTTPException(404, "Source not found")
    st["sources"] = keep
    return core.snapshot(st)


@app.post("/api/import")
def import_state(body: dict = Body(...)):
    """Body is a grid document: {sources, demand_mw, current_idx|index, weather}."""
    if not isinstance(body.get("sources"), list):
        raise HTTPException(400, "Invalid state JSON: missing 'sources'")
    return core.snapshot(core.clean_state(body))


@app.get("/api/model-performance")
def model_performance():
    data = core.read_optional("benchmark_results.json")
    metadata = core.read_optional("model_metadata.json") or {}
    production = {
        "demand": metadata.get("demand", {}).get("production_model"),
        "solar": metadata.get("solar", {}).get("production_model"),
        "wind": metadata.get("wind", {}).get("production_model"),
        "selected_on": "validation partition (test data never used for selection)",
        "trained_at_utc": metadata.get("trained_at_utc"),
    }
    if not data or not isinstance(data.get("results"), list):
        return {"available": False, "production": production,
                "message": "No benchmark results yet - run `python test.py` in the project folder, "
                           "then `python export_runtime.py`."}

    results = [r for r in data["results"] if isinstance(r, dict)]
    ranked = sorted([r for r in results if r.get("rank")], key=lambda r: r["rank"])
    shown = ranked[:6]
    best_ens = data.get("best_ensemble")
    if best_ens and not any(r.get("model") == best_ens.get("model") for r in shown):
        shown.append(best_ens)
    nb = next((r for r in results if r.get("type") == "classifier baseline"), None)
    if nb:
        shown.append(nb)
    return {
        "available": True,
        "generated_at_utc": data.get("generated_at_utc"), "mode": data.get("mode"),
        "tolerance": data.get("tolerance"), "tolerance_label": data.get("tolerance_label"),
        "ranking_rule": data.get("ranking_rule"),
        "derived_metrics_note": data.get("derived_metrics_note"),
        "demand_class_thresholds": data.get("demand_class_thresholds"),
        "dataset": data.get("dataset"), "split": data.get("split"), "approach": data.get("approach"),
        "production": production, "models": shown,
        "best_individual": data.get("best_individual"),
        "best_individual_reason": data.get("best_individual_reason"),
        "best_ensemble": best_ens, "best_ensemble_reason": data.get("best_ensemble_reason"),
        "not_applicable": data.get("not_applicable"),
    }


# Serve the frontend from FastAPI as well as the API.
# The Vercel function must include public/** in its bundled files.
@app.get("/", include_in_schema=False)
def index():
    if not (PUBLIC / "index.html").is_file():
        raise HTTPException(status_code=500, detail="Frontend index.html is missing from deployment")
    return FileResponse(PUBLIC / "index.html")


# Mount after API routes so it does not shadow them.
app.mount("/", StaticFiles(directory=str(PUBLIC), check_dir=False), name="public")