"""API smoke tests.  Run from the project folder:  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from fastapi.testclient import TestClient  # noqa: E402

from index import app  # noqa: E402

client = TestClient(app)
KEYS = ("sources", "demand_mw", "index", "weather", "history")


def state(snap):
    return {k: snap[k] for k in KEYS}


class ApiTests(unittest.TestCase):
    def test_default_snapshot(self):
        s = client.get("/api/state").json()
        self.assertEqual(len(s["sources"]), 7)
        self.assertGreaterEqual(s["dispatch"]["renewable_pct"], 0)

    def test_dispatch_balances_demand(self):
        d = client.get("/api/state").json()["dispatch"]
        supplied = d["renewable_used"] + d["fossil_used"] + d["unmet"]
        self.assertAlmostEqual(supplied, d["demand"], places=6)

    def test_time_step_moves_clock_and_history(self):
        s = client.get("/api/state").json()
        s2 = client.post("/api/time", json={"hours": 1, "state": state(s)}).json()
        self.assertNotEqual(s["time"], s2["time"])
        self.assertEqual(len(s2["history"]), 2)

    def test_demand_and_sources(self):
        s = client.get("/api/state").json()
        s = client.post("/api/demand", json={"set": 250, "state": state(s)}).json()
        self.assertEqual(s["demand_mw"], 250)
        self.assertGreater(s["dispatch"]["unmet"], 0)
        s = client.post("/api/sources", json={"type_id": 5, "capacity": 200, "state": state(s)}).json()
        self.assertEqual(s["dispatch"]["unmet"], 0)
        self.assertEqual(client.post("/api/sources", json={"type_id": 9, "capacity": 5}).status_code, 400)

    def test_battery_not_moved_by_refresh(self):
        s = client.get("/api/state").json()
        soc = [x["soc"] for x in s["sources"] if x["storage"]]
        s2 = client.post("/api/state", json={"state": state(s)}).json()
        self.assertEqual(soc, [x["soc"] for x in s2["sources"] if x["storage"]])

    def test_model_performance(self):
        self.assertTrue(client.get("/api/model-performance").json()["available"])


if __name__ == "__main__":
    unittest.main()
