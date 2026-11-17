"""Run parameters for every engine. The defaults are synthetic. SIMKIT_PARAMS=<json file> overrides them at run time;
an override file is a local input and is never committed or used in public CI (public logs and artifacts are world-readable)."""
import csv
import json
import os

import numpy as np

DEFAULTS = {
    "hc_income": 1.0,          # yearly income as a multiple of starting wealth (human capital)
    "hc_years": 10,
    "event_pool": None,        # CSV with a 'net' column of per-event returns; None = synthetic Student-t pool
    "event_pool_size": 400,
    "calib": None,             # JSON calibration for the calibrated world; None = that world is unavailable
    "cur_slots": 4,            # incumbent sizer: open slots, per-row cap, explore budget
    "cur_row_cap": 0.20,
    "explore": 0.20,
    "e_grid": None,            # alive-edge grid for the state filter; None = the engine's own grid
}


def _load():
    p = dict(DEFAULTS)
    path = os.environ.get("SIMKIT_PARAMS")
    if path:
        with open(path) as f:
            over = json.load(f)
        unknown = sorted(set(over) - set(DEFAULTS))
        if unknown:
            raise KeyError(f"unknown SIMKIT_PARAMS keys: {unknown}")
        p.update(over)
    return p


P = _load()


def event_pool():
    if P["event_pool"] is None:
        rng = np.random.default_rng(20261117)
        return 0.05 + 0.15 * rng.standard_t(3, P["event_pool_size"])
    with open(P["event_pool"]) as f:
        return np.array([float(r["net"]) for r in csv.DictReader(f)])


def calib():
    if P["calib"] is None:
        raise RuntimeError("the calibrated world needs SIMKIT_PARAMS with a calib file")
    with open(P["calib"]) as f:
        return json.load(f)


def e_grid(default):
    return default if P["e_grid"] is None else np.array(P["e_grid"], float)
