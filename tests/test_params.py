import json
import os
import subprocess
import sys

import numpy as np

import simkit_params as SP

SIMKIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simkit")


def _probe(code, params=None, tmp=None):
    env = {k: v for k, v in os.environ.items() if k != "SIMKIT_PARAMS"}
    if params is not None:
        p = os.path.join(tmp, "params.json")
        with open(p, "w") as f:
            json.dump(params, f)
        env["SIMKIT_PARAMS"] = p
    r = subprocess.run([sys.executable, "-c", code], cwd=SIMKIT, env=env, capture_output=True, text=True)
    return r


def test_defaults_are_synthetic():
    assert SP.P == SP.DEFAULTS
    a, b = SP.event_pool(), SP.event_pool()
    assert a.size == SP.DEFAULTS["event_pool_size"] and np.array_equal(a, b)


def test_override_reaches_the_engines(tmp_path):
    pool = tmp_path / "pool.csv"
    pool.write_text("net\n0.5\n-0.1\n0.2\n")
    r = _probe("import b2_engine as E, lifecycle_engine as L; print(E.Y, L.EVENT_NET.tolist())",
               {"hc_income": 3.5, "event_pool": str(pool)}, str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == "3.5" and "0.5, -0.1, 0.2" in r.stdout


def test_unknown_key_is_refused(tmp_path):
    r = _probe("import simkit_params", {"hc_incom": 1.0}, str(tmp_path))
    assert r.returncode != 0 and "unknown SIMKIT_PARAMS keys" in r.stderr


def test_calibrated_world_needs_a_file():
    try:
        SP.calib()
    except RuntimeError as e:
        assert "calib" in str(e)
    else:
        raise AssertionError("calib() must refuse without a file")


def test_cell_runs_on_synthetic_world(tmp_path):
    r = subprocess.run([sys.executable, "run_cell.py", "base", "1", "16", "CUR,NEW11", str(tmp_path)], cwd=SIMKIT,
                       env={k: v for k, v in os.environ.items() if k != "SIMKIT_PARAMS"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "base_1_16.npz")
    assert z["CUR"].shape == (16,) and np.isfinite(z["CUR"]).all() and np.isfinite(z["NEW11"]).all()
