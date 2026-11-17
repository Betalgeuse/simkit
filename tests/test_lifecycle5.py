

import numpy as np
import b2_engine as E
import lifecycle5_engine as A5


def _case(P=200, N=12, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.002, 0.004, (P, N)) * scale
    var = rng.uniform(1e-6, 2e-5, (P, N))
    so2 = np.full((P, N), E.S ** 2)
    fam = np.arange(N) // 4
    return rng, mean, var, so2, fam


def test_closed_form_when_uncoupled():
    _, mean, var, so2, fam = _case()
    P, N = mean.shape
    W = np.full(P, 1e4); kap = np.full(P, 2.0 / 60); sized = np.ones((P, N), bool)
    saved = A5.RHO_FAM; A5.RHO_FAM = 0.0
    try:
        x = A5.allocate(mean, var, so2, np.zeros((P, N)), fam, sized, W, kap, 0.5, np.c_[W, np.zeros((P, N))], 3000)
    finally:
        A5.RHO_FAM = saved
    want = np.maximum(0.5 * mean / (kap[:, None] * (so2 + var)), 0)
    assert np.abs(x[:, 1:] - want).max() < 1e-6 * want.max(), "uncoupled weights = kappa mu / (kap sigma^2) (Paleologo p.81)"


def test_capital_binding_ranks_by_mu_not_position():
    rng, mean, var, so2, fam = _case(scale=20)
    P, N = mean.shape
    W = np.ones(P); kap = np.full(P, 2.0); sized = np.ones((P, N), bool)
    perm = rng.permutation(N)
    fam_same = np.zeros(N, int)
    saved = A5.RHO_FAM; A5.RHO_FAM = 0.0
    try:
        x1 = A5.allocate(mean, var, so2, np.zeros((P, N)), fam_same, sized, W, kap, 1.0, np.c_[W, np.zeros((P, N))], 2000)
        x2 = A5.allocate(mean[:, perm], var[:, perm], so2, np.zeros((P, N)), fam_same, sized, W, kap, 1.0,
                         np.c_[W, np.zeros((P, N))], 2000)
    finally:
        A5.RHO_FAM = saved
    assert (x1.sum(1) <= W + 1e-9).all(), "sum <= W"
    assert np.abs(x1[:, 1:][:, perm] - x2[:, 1:]).max() < 1e-6, "weights follow the row, not its position (no arrival/hash order)"
    assert (x1.sum(1) > W - 1e-6).mean() > 0.9, "capital binds in this case"
    marg = mean
    kept = x1[:, 1:] > 1e-9
    dropped_pos = ~kept & (mean > 0)
    assert (np.where(dropped_pos, marg, -np.inf).max(1) <= np.where(kept, marg, np.inf).min(1) + 1e-12).all(), \
        "a dropped row never beats a kept row on mu (marginal CER of the first unit, KKT)"


def test_family_block_lowers_family_total():
    _, mean, var, so2, fam = _case(seed=3)
    P, N = mean.shape
    W = np.full(P, 50.0); kap = np.full(P, 2.0 / 60); sized = np.ones((P, N), bool)
    var = var * 50
    args = (mean, var, so2, np.zeros((P, N)), fam, sized, W, kap, 1.0, np.c_[W, np.zeros((P, N))], 3000)
    x_blk = A5.allocate(*args)
    saved = A5.RHO_FAM; A5.RHO_FAM = 0.0
    try:
        x_no = A5.allocate(*args[:4], fam, sized, W, kap, 1.0, np.c_[W, np.zeros((P, N))], 3000)
    finally:
        A5.RHO_FAM = saved
    assert x_blk[:, 1:].sum() < x_no[:, 1:].sum(), "shared family uncertainty shrinks the family's total"


def test_tz_delta_limits():
    _, mean, var, so2, fam = _case()
    P, N = mean.shape
    kap = np.full(P, 2.0 / 60); sized = np.ones((P, N), bool)
    x = np.c_[np.ones(P), np.abs(mean) * 10]
    _, d0 = A5.tz_blend(x, mean, np.full_like(var, 1e-30), so2, sized, kap)
    assert (d0 > 0.999).all(), "no estimation error -> keep the estimated rule (delta 1)"
    _, d1 = A5.tz_blend(x, mean, np.full_like(var, 1e3), so2, sized, kap)
    assert (d1 < 0.01).all(), "huge estimation error -> equal weights (delta 0)"
    xb, _ = A5.tz_blend(x, mean, var, so2, sized, kap)
    assert np.allclose(xb[:, 1:].sum(1), x[:, 1:].sum(1)), "blend keeps the satellite total"


def test_cur_operators():
    rng = np.random.default_rng(1)
    P, N = 300, 20
    sized = rng.random((P, N)) < 0.8
    held = sized & (rng.random((P, N)) < 0.3)
    arrive = rng.integers(0, 60, (P, N))
    s6 = A5.cur_slots(sized, held, arrive)
    assert (s6.sum(1) <= A5.CUR_SLOTS).all() and (s6 <= sized).all(), "at most CUR_SLOTS sized rows"
    nh = held.sum(1)
    assert ((s6 & held).sum(1) == np.minimum(nh, A5.CUR_SLOTS)).all(), "held rows keep their slot first"
    hold = rng.uniform(0, 0.05, (P, N)); nr = rng.random((P, N)) < 0.5
    order = np.argsort(rng.random((P, N)), 1); W = np.ones(P)
    g = A5.cur_greedy_explore(hold, nr, order, W)
    assert ((g * nr).sum(1) <= A5.EXPLORE + 1e-12).all(), "no-replay total <= EXPLORE * W"
    assert np.array_equal(g[~nr], hold[~nr]), "replay rows untouched"
    assert (g <= hold + 1e-15).all(), "freed room is not redistributed"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
