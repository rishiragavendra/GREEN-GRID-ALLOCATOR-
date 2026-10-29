"""
Green Grid Allocator
FastAPI Backend + Vercel Deployment

Supports:
- Smart grid simulation
- Energy demand management
- Weather simulation
- Energy source management
- ML predictions
- Model performance
- Static frontend
"""

import sys
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(API_DIR))

import _core as core

ROOT = core.ROOT
PUBLIC = ROOT / "public"

# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Green Grid Allocator",
    description="Intelligent Sustainable Energy Management API",
    version="1.0.0"
)

# --------------------------------------------------
# VERCEL ROUTING COMPATIBILITY
# --------------------------------------------------

@app.middleware("http")
async def vercel_routing_middleware(request: Request, call_next):

    path = request.scope["path"]

    if path in ("/api/index", "/api/index.py"):

        original_path = request.query_params.get("path")

        if original_path:

            original_path = original_path.lstrip("/")

            if original_path.startswith("api/"):
                request.scope["path"] = "/" + original_path
            else:
                request.scope["path"] = "/api/" + original_path

            request.scope["raw_path"] = (
                request.scope["path"].encode("utf-8")
            )

    return await call_next(request)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def _state(body):
    return core.clean_state((body or {}).get("state"))


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "ok": True
    }


# --------------------------------------------------
# GRID STATE
# --------------------------------------------------

@app.get("/api/state")
def get_state():

    return core.snapshot(
        core.default_state()
    )


@app.post("/api/state")
def refresh_state(
    body: dict = Body(default=None)
):

    return core.snapshot(
        _state(body)
    )


# --------------------------------------------------
# TIME MANAGEMENT
# --------------------------------------------------

@app.post("/api/time")
def set_time(
    body: dict = Body(...)
):

    st = _state(body)

    n = len(
        core.weather_table()["times"]
    )

    if body.get("datetime"):

        st["index"] = core.index_for_datetime(
            body["datetime"]
        )

    else:

        st["index"] = (
            st["index"] +
            int(body.get("hours", 1))
        ) % n

    st["weather"] = core.weather_at(
        st["index"]
    )

    return core.snapshot(
        st,
        commit=True
    )


# --------------------------------------------------
# ENERGY DEMAND
# --------------------------------------------------

@app.post("/api/demand")
def set_demand(
    body: dict = Body(...)
):

    st = _state(body)

    if body.get("set") is not None:

        st["demand_mw"] = max(
            0.0,
            float(body["set"])
        )

    elif body.get("delta"):

        st["demand_mw"] = max(
            0.0,
            st["demand_mw"] +
            float(body["delta"])
        )

    return core.snapshot(st)


# --------------------------------------------------
# WEATHER MANAGEMENT
# --------------------------------------------------

@app.post("/api/weather")
def set_weather(
    body: dict = Body(...)
):

    st = _state(body)

    for k, v in body.items():

        if (
            k in core.WEATHER_COLS
            and v is not None
            and v != ""
        ):

            try:

                st["weather"][k] = float(v)

            except (TypeError, ValueError):
                continue

    return core.snapshot(st)


# --------------------------------------------------
# ENERGY SOURCES
# --------------------------------------------------

@app.post("/api/sources")
def add_source(
    body: dict = Body(...)
):

    st = _state(body)

    try:

        tid = int(
            body.get("type_id", 0)
        )

        cap = float(
            body.get("capacity", 0)
        )

    except (TypeError, ValueError):

        raise HTTPException(
            status_code=400,
            detail="type_id and capacity must be numbers"
        )

    if not (
        1 <= tid <= len(core.SOURCE_TYPES)
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid source type"
        )

    if not (
        cap == cap
        and 0 < cap < float("inf")
    ):

        raise HTTPException(
            status_code=400,
            detail="Capacity must be positive"
        )

    new_id = max(
        [
            s.get("id", 0)
            for s in st["sources"]
        ],
        default=-1
    ) + 1

    st["sources"].append(
        core.make_source(
            tid,
            cap,
            new_id
        )
    )

    return core.snapshot(st)


# --------------------------------------------------
# REMOVE ENERGY SOURCE
# --------------------------------------------------

@app.post("/api/sources/remove")
def remove_source(
    body: dict = Body(...)
):

    st = _state(body)

    try:

        sid = int(
            body.get("id", -1)
        )

    except (TypeError, ValueError):

        raise HTTPException(
            status_code=400,
            detail="id must be a number"
        )

    keep = [
        s
        for s in st["sources"]
        if s.get("id") != sid
    ]

    if len(keep) == len(st["sources"]):

        raise HTTPException(
            status_code=404,
            detail="Source not found"
        )

    st["sources"] = keep

    return core.snapshot(st)


# --------------------------------------------------
# IMPORT GRID STATE
# --------------------------------------------------

@app.post("/api/import")
def import_state(
    body: dict = Body(...)
):

    if not isinstance(
        body.get("sources"),
        list
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid state JSON: missing 'sources'"
        )

    return core.snapshot(
        core.clean_state(body)
    )


# --------------------------------------------------
# MACHINE LEARNING MODEL PERFORMANCE
# --------------------------------------------------

@app.get("/api/model-performance")
def model_performance():

    data = core.read_optional(
        "benchmark_results.json"
    )

    metadata = (
        core.read_optional(
            "model_metadata.json"
        ) or {}
    )

    production = {

        "demand": metadata.get(
            "demand", {}
        ).get("production_model"),

        "solar": metadata.get(
            "solar", {}
        ).get("production_model"),

        "wind": metadata.get(
            "wind", {}
        ).get("production_model"),

        "selected_on":
            "validation partition (test data never used for selection)",

        "trained_at_utc":
            metadata.get("trained_at_utc")
    }

    if (
        not data
        or not isinstance(
            data.get("results"),
            list
        )
    ):

        return {

            "available": False,

            "production": production,

            "message":
                "Benchmark results unavailable. "
                "Run python test.py and "
                "python export_runtime.py."
        }

    results = [
        r
        for r in data["results"]
        if isinstance(r, dict)
    ]

    ranked = sorted(
        [
            r
            for r in results
            if r.get("rank")
        ],
        key=lambda r: r["rank"]
    )

    shown = ranked[:6]

    best_ens = data.get(
        "best_ensemble"
    )

    if best_ens and not any(
        r.get("model") == best_ens.get("model")
        for r in shown
    ):

        shown.append(best_ens)

    nb = next(
        (
            r
            for r in results
            if r.get("type") == "classifier baseline"
        ),
        None
    )

    if nb:
        shown.append(nb)

    return {

        "available": True,

        "generated_at_utc":
            data.get("generated_at_utc"),

        "mode":
            data.get("mode"),

        "tolerance":
            data.get("tolerance"),

        "tolerance_label":
            data.get("tolerance_label"),

        "ranking_rule":
            data.get("ranking_rule"),

        "derived_metrics_note":
            data.get("derived_metrics_note"),

        "demand_class_thresholds":
            data.get("demand_class_thresholds"),

        "dataset":
            data.get("dataset"),

        "split":
            data.get("split"),

        "approach":
            data.get("approach"),

        "production":
            production,

        "models":
            shown,

        "best_individual":
            data.get("best_individual"),

        "best_individual_reason":
            data.get("best_individual_reason"),

        "best_ensemble":
            best_ens,

        "best_ensemble_reason":
            data.get("best_ensemble_reason"),

        "not_applicable":
            data.get("not_applicable")
    }


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

@app.get("/", include_in_schema=False)
def index():

    index_file = PUBLIC / "index.html"

    if not index_file.is_file():

        raise HTTPException(
            status_code=500,
            detail="Frontend index.html missing"
        )

    return FileResponse(index_file)


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

# Keep this mount after all API routes.

app.mount(
    "/",
    StaticFiles(
        directory=str(PUBLIC),
        check_dir=False
    ),
    name="public"
)