import json
import math
import numpy as np
import pandas as pd  # type: ignore[import-not-found]
from pathlib import Path

import ml_pipeline as mlp

# Paths are resolved from this file (via ml_pipeline) so the simulator works
# no matter which directory it is launched from.
ROOT = mlp.ROOT
MODEL_DIR = mlp.MODEL_DIR
DATA_PATH = mlp.DATA_PATH
STATE_PATH = ROOT / "grid_state.json"

# Re-exported from the shared pipeline so the feature definitions used at
# inference time can never drift away from the ones used during training.
WEATHER_COLS = mlp.WEATHER_COLS
ZONE_COLS = mlp.ZONE_COLS

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

BATTERY_EFFICIENCY = 0.9  # round-trip: 90% of stored energy is recoverable


def _jsonable(o):
    """Convert numpy scalars to plain Python types so saved state stays numeric."""
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)

class GridSimulator:
    def __init__(self):
        self.load_artifacts()
        self.load_data()
        self.reset_grid()
        
    def load_artifacts(self):
        """Load the production models trained by `python train.py`.

        Raises FileNotFoundError with actionable instructions if the artifacts
        are missing, rather than failing with an opaque traceback.
        """
        bundle = mlp.load_production_bundle()
        self.demand_model = bundle["demand_model"]
        self.solar_model = bundle["solar_model"]
        self.wind_model = bundle["wind_model"]
        self.scaler_demand = bundle["scaler_demand"]
        self.scaler_solar = bundle["scaler_solar"]
        self.scaler_wind = bundle["scaler_wind"]
        self.features = bundle["features"]
        self.last_weather = bundle["last_weather"]
        self.model_metadata = bundle["metadata"]

    @property
    def model_names(self):
        """Which algorithm currently backs each forecasting target."""
        md = getattr(self, "model_metadata", {}) or {}
        return {
            "demand": md.get("demand", {}).get("production_model", "unknown"),
            "solar": md.get("solar", {}).get("production_model", "unknown"),
            "wind": md.get("wind", {}).get("production_model", "unknown"),
        }

    def load_data(self):
        # Identical cleaning/resampling path as training (ml_pipeline).
        df = mlp.load_and_clean(DATA_PATH)
        self.full_df = df  # keep everything: real demand + weather for true lag features
        self.weather_df = df[WEATHER_COLS].copy()
        self.current_idx = len(self.weather_df) - 1
        
    def reset_grid(self):
        self.sources = []
        for i, s in enumerate(DEFAULT_SOURCES):
            st = SOURCE_TYPES[s["type_id"] - 1]
            self.sources.append({
                "id": i,
                "type_id": s["type_id"],
                "name": st["name"],
                "icon": st["icon"],
                "capacity": s["capacity_mw"],
                "available": 0.0,
                "used": 0.0,
                "emission": st["emission"],
                "cost": st["cost"],
                "renewable": st["renewable"],
                "storage": st.get("storage", False),
                "soc": s.get("soc", 0.0),
                "max_charge": s["capacity_mw"],
                "max_discharge": s["capacity_mw"],
            })
        self.demand_mw = 100.0  # static demand - adjust with +/- 
        self.current_time = self.weather_df.index[self.current_idx]
        self.update_weather_from_data()
        
    def update_weather_from_data(self):
        if self.current_idx < len(self.weather_df):
            row = self.weather_df.iloc[self.current_idx]
            self.weather = row.to_dict()
            self.current_time = self.weather_df.index[self.current_idx]
        else:
            self.weather = self.last_weather
            
    def get_time_features(self):
        """Cyclical/calendar features - shared with training via ml_pipeline."""
        return mlp.time_features(self.current_time)


    def predict_demand(self):
        tf = self.get_time_features()
        features = {**self.weather, **tf}

        # TRUE historical lags from the real data (same semantics as training):
        # lag24/lag168 = value N hours before now; roll24/roll168 = mean of the
        # last N hours strictly before the current hour (no look-ahead).
        idx = self.current_idx
        for col in ZONE_COLS:
            hist = self.full_df[col].iloc[max(0, idx - 168):idx].tolist()
            for lag in (24, 168):
                features[f"{col}_lag{lag}"] = hist[-lag] if len(hist) >= lag else (hist[-1] if hist else 30000)
            features[f"{col}_roll24"] = float(np.mean(hist[-24:])) if hist else 30000
            features[f"{col}_roll168"] = float(np.mean(hist[-168:])) if hist else 30000

        # TRUE weather lags from the real data too
        for wcol in WEATHER_COLS:
            wh = self.full_df[wcol].iloc[max(0, idx - 168):idx].tolist()
            for lag in (24, 168):
                features[f"{wcol}_lag{lag}"] = wh[-lag] if len(wh) >= lag else (wh[-1] if wh else self.weather[wcol])
            features[f"{wcol}_roll24"] = float(np.mean(wh[-24:])) if wh else self.weather[wcol]

        # Create feature vector
        feat_names = self.features["demand"]
        X = np.array([[features.get(f, 0) for f in feat_names]])
        X_scaled = self.scaler_demand.transform(X)
        demand = self.demand_model.predict(X_scaled)[0]
        return max(0, demand / 1000)  # Convert to MW
        
    def predict_solar_cf(self):
        tf = self.get_time_features()
        features = {**self.weather, **tf}
        feat_names = self.features["solar"]
        X = np.array([[features.get(f, 0) for f in feat_names]])
        X_scaled = self.scaler_solar.transform(X)
        cf = self.solar_model.predict(X_scaled)[0]
        return np.clip(cf, 0, 1)
        
    def predict_wind_cf(self):
        tf = self.get_time_features()
        features = {**self.weather, **tf}
        feat_names = self.features["wind"]
        X = np.array([[features.get(f, 0) for f in feat_names]])
        X_scaled = self.scaler_wind.transform(X)
        cf = self.wind_model.predict(X_scaled)[0]
        return np.clip(cf, 0, 1)
        
    def compute_availability(self):
        solar_cf = self.predict_solar_cf()
        wind_cf = self.predict_wind_cf()

        for s in self.sources:
            if s["type_id"] == 1:  # Solar
                s["available"] = s["capacity"] * solar_cf
            elif s["type_id"] == 2:  # Wind
                s["available"] = s["capacity"] * wind_cf
            elif s["type_id"] == 3:  # Hydro
                s["available"] = s["capacity"] * 0.9
            elif s["type_id"] == 4:  # Battery
                s["available"] = s["soc"] * s["capacity"]
            else:  # Fossil
                s["available"] = s["capacity"]

    def allocate(self):
        self.compute_availability()

        # Merit Order Dispatch with working storage:
        # Phase 1: Renewables (Solar, Wind, Hydro) run at FULL output (no capping),
        #          so surplus energy is captured instead of silently lost.
        # Phase 2: Battery - charge with surplus OR discharge into a deficit.
        # Phase 3: Fossil (Gas, Coal, Diesel) - covers whatever is still missing.

        for s in self.sources:
            s["used"] = 0.0

        remaining = self.demand_mw
        total_renewable = 0.0
        total_fossil = 0.0
        co2_total = 0.0
        battery_charged = 0.0
        battery_discharged = 0.0
        curtailed = 0.0

        # Phase 1: full renewable output
        for s in self.sources:
            if s["renewable"] and not s["storage"]:
                s["used"] = s["available"]
                total_renewable += s["available"]

        surplus = max(0.0, total_renewable - self.demand_mw)
        deficit = max(0.0, self.demand_mw - total_renewable)

        # Phase 2: battery - charge surplus or cover deficit
        for s in self.sources:
            if not s["storage"]:
                continue
            if surplus > 0:
                headroom = (1.0 - s["soc"]) * s["capacity"]
                charge = min(surplus, headroom, s.get("max_charge", s["capacity"]))
                if charge > 0:
                    s["used"] = -charge  # Negative = charging
                    s["soc"] = min(1.0, s["soc"] + (charge * BATTERY_EFFICIENCY) / s["capacity"])
                    battery_charged += charge
                    surplus -= charge
                curtailed = surplus  # energy the battery can't absorb is curtailed
            elif deficit > 0:
                discharge = min(deficit, s["soc"] * s["capacity"], s.get("max_discharge", s["capacity"]))
                if discharge > 0:
                    s["used"] = discharge
                    s["soc"] = max(0.0, s["soc"] - discharge / s["capacity"])
                    battery_discharged += discharge
                    deficit -= discharge
                    total_renewable += discharge

        remaining = deficit

        # Phase 3: fossil - by cost priority
        fossil_sources = [s for s in self.sources if not s["renewable"]]
        fossil_sources.sort(key=lambda x: x["cost"])
        for s in fossil_sources:
            used = min(s["available"], remaining)
            s["used"] = used
            total_fossil += used
            co2_total += used * s["emission"]
            remaining -= used
            if remaining <= 0:
                break

        unmet = max(0, remaining)
        return {
            "demand": self.demand_mw,
            # renewables actually consumed (capped at demand; the rest is curtailed)
            "renewable_used": min(total_renewable, self.demand_mw),
            "renewable_generated": total_renewable,
            "fossil_used": total_fossil,
            "battery_charged": battery_charged,
            "battery_discharged": battery_discharged,
            "curtailed": curtailed,
            "unmet": unmet,
            "co2": co2_total,
            "renewable_pct": (min(total_renewable, self.demand_mw) / self.demand_mw * 100) if self.demand_mw > 0 else 0
        }
        
    def render(self):
        result = self.allocate()
        
        print("\n" + "=" * 72)
        print(f"  GREEN GRID ALLOCATOR  |  Weather: {self.weather['Temperature']:.1f}C, {self.weather['Humidity']:.0f}% RH, "
              f"{self.weather['Wind Speed']:.1f} m/s wind  |  Time: {self.current_time.strftime('%H:%M %a %d %b')}")
        print("=" * 72)
        print(f"  {'SOURCE':<20} {'CAP':>5} {'AVAIL':>6} {'USED':>6} {'TYPE':<12} {'BAR'}")
        print("  " + "-" * 70)
        
        for s in self.sources:
            cap = f"{s['capacity']:.0f}"
            avail = f"{s['available']:.1f}"
            used = f"{s['used']:.1f}"
            type_str = "Renewable" if s["renewable"] else "Fossil"
            if s["storage"]:
                type_str = "Storage"
                used = f"{s['used']:.1f}"
                if s["used"] < 0:
                    used = f"chg {abs(s['used']):.1f}"
                    
            # Bar visualization
            if s["capacity"] > 0:
                pct = s["used"] / s["capacity"] if not s["storage"] else (s["soc"])
                if s["storage"]:
                    pct = s["soc"]
                bar_len = int(pct * 20)
                if s["renewable"] and not s["storage"]:
                    bar = "#" * bar_len + "-" * (20 - bar_len)
                elif s["storage"]:
                    bar = "=" * bar_len + "-" * (20 - bar_len)
                else:
                    bar = "+" * bar_len + "-" * (20 - bar_len)
            else:
                bar = "-" * 20
                
            icon = s["icon"]
            print(f"  {icon} {s['name']:<15} {cap:>5} {avail:>6} {used:>6} {type_str:<12} {bar}")
            
        print("  " + "-" * 70)
        print(f"  DEMAND: {result['demand']:.1f} MW  |  RENEWABLE: {result['renewable_used']:.1f} MW ({result['renewable_pct']:.1f}%)  |  "
              f"FOSSIL: {result['fossil_used']:.1f} MW")
        battery = next((s for s in self.sources if s["storage"]), None)
        if result['battery_charged'] > 0:
            batt_status = f"Charging {result['battery_charged']:.1f} MW"
        elif result['battery_discharged'] > 0:
            batt_status = f"Discharging {result['battery_discharged']:.1f} MW"
        else:
            batt_status = "Idle"
        batt_soc = f"  |  SoC: {battery['soc']*100:.0f}%" if battery else ""
        print(f"  CO2: {result['co2']:.1f} t/h  |  Battery: {batt_status}{batt_soc}")
        if result['curtailed'] > 0:
            print(f"  CURTAILED RENEWABLES: {result['curtailed']:.1f} MW")
        if result['unmet'] > 0:
            print(f"  UNMET DEMAND: {result['unmet']:.1f} MW")
        print("=" * 72)
        print("  Commands: [a]dd  [r]emove  [+]demand  [-]demand  [w]eather  [t]ime  [s]ave  [l]oad  [i]mport  [e]xport  [q]uit")
        print("=" * 72)
        
    def add_source_by_type(self, tid, capacity):
        """Add a source programmatically (CLI + web UI). Returns the new source or None."""
        if tid < 1 or tid > len(SOURCE_TYPES):
            return None
        try:
            if not (math.isfinite(capacity) and capacity > 0):
                return None
        except TypeError:
            return None
        st = SOURCE_TYPES[tid - 1]
        new_id = max([s.get("id", 0) for s in self.sources]) + 1 if self.sources else 0
        src = {
            "id": new_id,
            "type_id": tid,
            "name": st["name"],
            "icon": st["icon"],
            "capacity": capacity,
            "available": 0.0,
            "used": 0.0,
            "emission": st["emission"],
            "cost": st["cost"],
            "renewable": st["renewable"],
            "storage": st.get("storage", False),
            "soc": 0.5 if st.get("storage", False) else 0.0,
            "max_charge": capacity,
            "max_discharge": capacity,
        }
        self.sources.append(src)
        return src

    def remove_source_by_id(self, src_id):
        """Remove a source by its id. Returns True if removed."""
        for i, s in enumerate(self.sources):
            if s["id"] == src_id:
                self.sources.pop(i)
                return True
        return False

    def add_source(self):
        print("\nSource Types:")
        for st in SOURCE_TYPES:
            print(f"  {st['id']}. {st['icon']} {st['name']} (emission: {st['emission']} kg/kWh)")
        try:
            tid = int(input("Select type (1-7): "))
            cap = float(input("Capacity (MW): "))
            src = self.add_source_by_type(tid, cap)
            if src:
                print(f"Added {src['icon']} {src['name']} ({cap} MW)")
            else:
                print("Invalid type or capacity")
        except ValueError:
            print("Invalid input")
            
    def remove_source(self):
        if not self.sources:
            print("No sources to remove")
            return
        print("\nCurrent sources:")
        for i, s in enumerate(self.sources):
            print(f"  {i}. {s['icon']} {s['name']} - {s['capacity']} MW")
        try:
            idx = int(input("Select index to remove: "))
            if 0 <= idx < len(self.sources):
                removed = self.sources.pop(idx)
                print(f"Removed {removed['icon']} {removed['name']}")
            else:
                print("Invalid index")
        except ValueError:
            print("Invalid input")
            
    def change_demand(self, delta):
        self.demand_mw = max(0, self.demand_mw + delta)
        print(f"Demand set to {self.demand_mw:.1f} MW")
        
    def set_weather(self):
        print(f"\nCurrent: {self.weather}")
        try:
            self.weather["Temperature"] = float(input(f"Temperature [{self.weather['Temperature']:.1f}]: ") or self.weather["Temperature"])
            self.weather["Humidity"] = float(input(f"Humidity [{self.weather['Humidity']:.1f}]: ") or self.weather["Humidity"])
            self.weather["Wind Speed"] = float(input(f"Wind Speed [{self.weather['Wind Speed']:.1f}]: ") or self.weather["Wind Speed"])
            self.weather["general diffuse flows"] = float(input(f"Gen Diffuse [{self.weather['general diffuse flows']:.3f}]: ") or self.weather["general diffuse flows"])
            self.weather["diffuse flows"] = float(input(f"Diffuse [{self.weather['diffuse flows']:.3f}]: ") or self.weather["diffuse flows"])
            print("Weather updated")
        except ValueError:
            print("Invalid input")
            
    def advance_time(self):
        self.current_idx = (self.current_idx + 1) % len(self.weather_df)
        self.update_weather_from_data()
        print(f"Time advanced to {self.current_time.strftime('%H:%M %a %d %b')}")
        
    def _apply_state(self, state):
        """Validate + apply a state dict (from MongoDB, JSON import or legacy file)."""
        if not isinstance(state, dict) or "sources" not in state:
            print("Invalid state: missing 'sources'")
            return False

        # Sanitize: older saves wrote numbers as strings ("0.0") - coerce them back
        def _coerce(obj):
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_coerce(v) for v in obj]
            if isinstance(obj, str):
                try:
                    return float(obj)
                except ValueError:
                    return obj
            return obj

        state = _coerce(state)
        self.sources = state["sources"]
        self.demand_mw = float(state.get("demand_mw", 100.0))
        self.current_idx = int(state.get("current_idx", 0)) % len(self.weather_df)
        if state.get("weather"):
            self.weather = state["weather"]
        self.current_time = self.weather_df.index[self.current_idx]
        return True

    def save_state(self):
        """Persist the current grid state to grid_state.json."""
        state = {
            "sources": self.sources,
            "demand_mw": self.demand_mw,
            "current_idx": self.current_idx,
            "weather": self.weather,
        }
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, default=_jsonable)
        print("State saved to grid_state.json")

    def load_state(self):
        """Load the grid state from grid_state.json. Returns True on success."""
        try:
            with open(STATE_PATH, "r") as f:
                state = json.load(f)
        except FileNotFoundError:
            print("No saved state found (grid_state.json)")
            return False
        except json.JSONDecodeError as e:
            print(f"Invalid state file ({e})")
            return False
        if self._apply_state(state):
            print("State loaded from grid_state.json")
            return True
        return False

    def export_json(self, filename="grid_export.json"):
        """Export the current grid state to a local JSON file."""
        state = {
            "sources": self.sources,
            "demand_mw": self.demand_mw,
            "current_idx": self.current_idx,
            "weather": self.weather,
        }
        with open(filename, "w") as f:
            json.dump(state, f, default=_jsonable, indent=2)
        print(f"State exported to {filename}")

    def import_json(self, filename):
        """Import a grid state from a local JSON file."""
        try:
            with open(filename, "r") as f:
                state = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return False
        except json.JSONDecodeError as e:
            print(f"Invalid JSON ({e})")
            return False
        if self._apply_state(state):
            print(f"State imported from {filename}")
            return True
        return False
            
    def run(self):
        print("\n" + "=" * 72)
        print("  GREEN GRID ALLOCATOR - Interactive Simulator")
        names = self.model_names
        print(f"  Forecast models - demand: {names['demand']} | "
              f"solar: {names['solar']} | wind: {names['wind']}")
        print("  Type 'h' for help, 'q' to quit")
        print("=" * 72)

        # Import the saved state on start (auto-resume)
        if self.load_state():
            print("  (resumed from grid_state.json)")
        else:
            print("  (no saved state - starting with the default grid)")

        while True:
            self.render()
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'q':
                self.save_state()  # auto-save on quit
                print("Goodbye!")
                break
            elif cmd == 'h':
                print("\nCommands:")
                print("  a - Add source")
                print("  r - Remove source")
                print("  + - Increase demand (+10 MW)")
                print("  - - Decrease demand (-10 MW)")
                print("  w - Set weather manually")
                print("  t - Advance time by 1 hour")
                print("  s - Save state to grid_state.json")
                print("  l - Load state from grid_state.json")
                print("  i - Import grid state from JSON file")
                print("  e - Export grid state to JSON file")
                print("  q - Quit (auto-saves to grid_state.json)")
            elif cmd == 'a':
                self.add_source()
            elif cmd == 'r':
                self.remove_source()
            elif cmd == '+':
                self.change_demand(10)
            elif cmd == '-':
                self.change_demand(-10)
            elif cmd == 'w':
                self.set_weather()
            elif cmd == 't':
                self.advance_time()
            elif cmd == 's':
                self.save_state()
            elif cmd == 'l':
                self.load_state()
            elif cmd == 'i':
                fn = input("JSON file to import [grid_import.json]: ").strip() or "grid_import.json"
                self.import_json(fn)
            elif cmd == 'e':
                fn = input("JSON file to export [grid_export.json]: ").strip() or "grid_export.json"
                self.export_json(fn)
            else:
                print("Unknown command. Type 'h' for help.")


def main():
    try:
        sim = GridSimulator()
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1)
    sim.run()


if __name__ == "__main__":
    main()