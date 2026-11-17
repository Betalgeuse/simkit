

import numpy as np
import b2_engine as E
import lifecycle_engine as L
import lifecycle5_engine as A5
import lifecycle6_engine as A6


def test_tz_delta_equals_prop2_literal():

    rng = np.random.default_rng(4)
    P, N, T, gam = 500, 10, 120, 3.0
    s2 = rng.uniform(0.02, 0.06, (P, N)) ** 2
    muhat = rng.normal(0.01, 0.02, (P, N))
    we = np.full((P, N), 1.0 / N)
    th2 = (muhat ** 2 / s2).sum(1) - N / T
    p1 = (we * we * s2).sum(1) - 2 / gam * (we * muhat).sum(1) + th2 / gam ** 2
    p2 = (N / T) / gam ** 2
    want = p1 / (p1 + p2)
    ok = p1 > 0
    got = A6.tz_delta(muhat / (gam * s2), we, s2 / T, s2, np.ones((P, N), bool), np.full(P, gam), 1.0)
    assert ok.mean() > 0.5 and np.allclose(got[ok], want[ok], rtol=1e-10), "delta = Tu-Zhou Prop. 2 (Eq. 9-10)"
    old = A5.tz_blend(np.c_[np.ones(P), muhat / (gam * s2)], muhat, s2 / T, s2 - s2 / T, np.ones((P, N), bool),
                      np.full(P, gam))[1]
    print(f"  L5 delta vs Prop. 2 on this case: median {np.median(old[ok]):.3f} vs {np.median(want[ok]):.3f}")


def test_tz_p1_unbiased_and_delta_minimizes_loss():

    rng = np.random.default_rng(5)
    N, R, kap, kappa = 8, 200_000, 2.0, 0.25
    s2 = rng.uniform(0.02, 0.06, N) ** 2
    mu = rng.normal(0.004, 0.004, N); var = s2 / 30
    target = kappa * mu / (kap * s2)
    e = np.full(N, target.sum() / N) * 0.6 + 0.01
    muhat = mu + rng.normal(0, 1, (R, N)) * np.sqrt(var)
    w = kappa * muhat / (kap * s2)
    p2 = (kappa / kap) ** 2 * (var / s2).sum()
    p1_hat = (s2 * (e - w) ** 2).sum(1) - p2
    p1 = (s2 * (e - target) ** 2).sum()
    assert abs(p1_hat.mean() - p1) < 4 * p1_hat.std() / np.sqrt(R), "p1^ is unbiased (Eq. 9)"
    loss = lambda dl: (s2 * ((1 - dl) * e + dl * w - target) ** 2).sum(1).mean()
    d_star = p1 / (p1 + p2)
    d_hat = A6.tz_delta(w[:2000], e[None, :], np.broadcast_to(var, (2000, N)), np.broadcast_to(s2, (2000, N)),
                        np.ones((2000, N), bool), np.full(2000, kap), kappa)
    assert abs(np.median(d_hat) - d_star) < 0.1, "tz_delta estimates delta* for the kappa-scaled rule"
    assert loss(d_star) <= min(loss(d_star - 0.05), loss(d_star + 0.05)), "delta* minimizes expected loss (Eq. 7)"


def test_fence_caps_satellites_and_ranks_by_mu():
    rng = np.random.default_rng(6)
    P, N = 300, 12
    mean = rng.normal(0.02, 0.03, (P, N)); var = np.full((P, N), 1e-6); so2 = np.full((P, N), E.S ** 2)
    fam = np.zeros(N, int); W = np.ones(P); kap = np.full(P, 2.0); sized = np.ones((P, N), bool)
    x0 = np.c_[W, np.zeros((P, N))]
    saved = A5.RHO_FAM; A5.RHO_FAM = 0.0
    try:
        free = A6.allocate(mean, var, so2, np.zeros((P, N)), fam, sized, W, kap, 1.0, x0, 2000)
        gb = np.full(P, 0.05)
        x = A6.allocate(mean, var, so2, np.zeros((P, N)), fam, sized, W, kap, 1.0, x0, 2000, gb)
        big = A6.allocate(mean, var, so2, np.zeros((P, N)), fam, sized, W, kap, 1.0, x0, 2000, np.full(P, 10.0))
    finally:
        A5.RHO_FAM = saved
    assert (x[:, 1:].sum(1) <= gb + 1e-9).all() and (x.sum(1) <= W + 1e-9).all(), "satellite sum <= fence room, total <= W"
    assert np.abs(big - free).max() < 1e-12, "a fence that does not bind changes nothing"
    kept = x[:, 1:] > 1e-9; dropped = ~kept & (mean > 0)
    assert (np.where(dropped, mean, -np.inf).max(1) <= np.where(kept, mean, np.inf).min(1) + 1e-12).all(), \
        "under the fence a dropped row never beats a kept row on mu"


def test_cur_fence_greedy_held_first():
    rng = np.random.default_rng(7)
    P, N = 300, 20
    hold = rng.uniform(0, 0.03, (P, N)); held = rng.random((P, N)) < 0.3
    order = np.argsort(rng.random((P, N)), 1); gb = rng.uniform(0, 0.3, P)
    g = A6.cur_fence(hold, held, order, gb)
    assert (g.sum(1) <= np.minimum(gb, hold.sum(1)) + 1e-12).all() and (g <= hold + 1e-15).all(), "sum <= room, no redistribution"
    hs = (hold * held).sum(1)
    full_held = hs <= gb
    assert np.allclose((g * held).sum(1)[full_held], hs[full_held]), "held rows are served before new entries"


def test_rare_long_run_variance_is_hold_times_monthly():
\

    rng = np.random.default_rng(8)
    P, M, H = 4000, 120, L.HOLD
    c = np.minimum(rng.poisson(L.RATE, (P, M + H)), L.MAXC)
    s1 = np.zeros((P, M + H))
    for j in range(L.MAXC):
        s1 += np.where(j < c, L.EVENT_NOISE[rng.integers(0, L.EVENT_NOISE.size, (P, M + H))] / H, 0.0)
    cs = np.cumsum(np.pad(c, ((0, 0), (1, 0))), 1); ss = np.cumsum(np.pad(s1, ((0, 0), (1, 0))), 1)
    t = np.arange(H, M + H)
    n_open = cs[:, t + 1] - cs[:, t + 1 - H]; s_open = ss[:, t + 1] - ss[:, t + 1 - H]
    ry = np.where(n_open > 0, s_open / np.maximum(n_open, 1), 0.0)
    so2 = (L.EVENT_SD / H) ** 2
    one_month = (so2 / np.maximum(n_open, 1)).mean()
    long_run = ry.sum(1).var() / M
    ratio = long_run / one_month
    print(f"  long-run / one-month variance = {ratio:.1f} (HOLD = {H})")
    assert 0.6 * H < ratio < 1.4 * H, "the long-run variance is ~HOLD x the one-month variance"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
