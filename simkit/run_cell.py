#!/usr/bin/env python3
"""One cell = one (world, seed, paths, arm). usage: run_cell.py WORLD SEED PATHS ARM[,ARM..] OUTDIR
Writes OUTDIR/<world>_<seed>_<paths>.npz (terminal wealth W and satellite share per arm) and a .json timing line."""
import json
import os
import sys
import time

import numpy as np

import b2_engine as E
import lifecycle_engine as L
import lifecycle2_engine as L2
import lifecycle7_engine as A7
import lifecycle11_engine as A11
import simkit_params as SP

W0 = E.World
WORLDS = {
    "zero_edge": lambda p, s: L.LWorld(W0(edge=0.0, paths=p, seed=s)),
    "base": lambda p, s: L.LWorld(W0(paths=p, seed=s)),
    "prior4x": lambda p, s: L.LWorld(W0(edge=0.0025, paths=p, seed=s)),
    "fat_tail": lambda p, s: L.LWorld(W0(edge=L.EVENT_EDGE, paths=p, seed=s), kind="fat"),
    "rare_slow": lambda p, s: L.LWorld(W0(mix=(1.0, 0.0, 0.0, 0.0), paths=p, seed=s), kind="rare"),
    "decay": lambda p, s: L.LWorld(W0(paths=p, seed=s), half_life=36),
}
ITERS = {"fat_tail": 200}


def cell(world, paths, seed):
    crisis = world == "crisis"
    name = "fat_tail" if crisis else world
    lw2 = L2.LWorld2(WORLDS[name](paths, seed))
    return lw2, L2.draw2(lw2), crisis, ITERS.get(name, 25)


def run_arm(arm, lw2, d, crisis, iters):
    if arm == "CUR":
        return A7.run7(lw2, A7.Arm7("CUR", "CUR"), d=d, iters=iters, crisis=crisis)
    if arm == "NEW025":
        return A7.run7(lw2, A7.Arm7("NEW", "NEW", 0.25), d=d, iters=iters, crisis=crisis)
    if arm == "NEW11":
        return A11.run8(lw2, d=d, iters=iters, crisis=crisis, e_grid=SP.e_grid(A11.E_GRID), s_scen=64, kink=E.KINK,
                        beta_var=True, alloc="new")
    raise SystemExit(f"unknown arm {arm}")


def main(world, seed, paths, arms, outdir):
    os.makedirs(outdir, exist_ok=True)
    lw2, d, crisis, iters = cell(world, paths, seed)
    out, timing = {}, {}
    for arm in arms:
        t0 = time.time()
        W, info = run_arm(arm, lw2, d, crisis, iters)
        timing[arm] = round(time.time() - t0, 2)
        out[arm] = W
        out[arm + "_sat"] = info["sat_share"]
        print(arm, f"sat {info['sat_share']:.3f} {timing[arm]:.1f}s", flush=True)
    stem = os.path.join(outdir, f"{world}_{seed}_{paths}")
    np.savez(stem + ".npz", **out)
    with open(stem + ".json", "w") as f:
        json.dump({"world": world, "seed": seed, "paths": paths, "seconds": timing}, f)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(","), sys.argv[5])
