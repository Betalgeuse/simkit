
\
\
\
\
\
\
\
\

import math
from dataclasses import dataclass
from statistics import NormalDist
import numpy as np
import core_calib as CC
import simkit_params as SP

G, YEARS = 2.0, 10
MONTHS = 12 * YEARS
RF = CC.RF
MU_C, SD_C = CC.MU_C, CC.SD_C
PC, CR = CC.PC, CC.CR
S, CAP, BOOK, HAIRCUT = 0.03, 0.25, 0.01, 0.42
Y, HC_YEARS = SP.P['hc_income'], SP.P['hc_years']
KINK = -0.04
SHAPES = {"indep": (0.0, 0.0), "linear": (0.3, 0.0), "put": (0.0, 0.49), "hedge": (0.0, -0.3)}
SHAPE_NAMES = list(SHAPES)
ND = NormalDist()
EULER = 0.5772156649


def _core_moments():

    rng = np.random.default_rng(0)
    r = CC.draw_excess(rng, 4_000_000)
    return CC.EX_TARGET, CC.VAR_TARGET, np.minimum(r - KINK, 0.0).mean()


def _assert_core():

    draw = lambda rng, n: np.exp(rng.normal(MU_C, SD_C, n)) - 1 + np.where(rng.random(n) < PC, CR, 0.0) - RF
    CC.assert_matches_stated(draw)


_assert_core()
EX_C, VC, KINK_MEAN = _core_moments()


@dataclass(frozen=True)
class World:
    n_rows: int = 20
    t0: int = 60
    edge: float = 0.005
    mix: tuple = (0.25, 0.25, 0.25, 0.25)
    p_real: float = 0.5
    p_die: float = 1 / 36
    p_revive: float = 1 / 48
    birth_window: int = 60
    paths: int = 2000
    seed: int = 7
    rho: float = 0.0
    sizer_rho: bool = True
    common_regime: bool = False
    put_k: float = 0.49


@dataclass(frozen=True)
class Arm:
    name: str
    rule: str = "all"
    thresh: float = 0.0
    drop_fakes: bool = False
    core_only: bool = False


def shape_ret(r_ex, a, k):
    return a * (r_ex - EX_C) + k * (np.minimum(r_ex - KINK, 0.0) - KINK_MEAN)


def psr(sr, n, skew, kurt, sr0=0.0):

    den = np.sqrt(np.maximum(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    z = (sr - sr0) * np.sqrt(np.maximum(n - 1, 1)) / den
    return 0.5 * (1 + _erf(z / math.sqrt(2)))


def _erf(x):

    sgn = np.sign(x); x = np.abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sgn * y


def moments(y, axis=-1):
    m = y.mean(axis)
    c = y - np.expand_dims(m, axis)
    v = (c ** 2).mean(axis)
    sd = np.sqrt(np.maximum(v, 1e-18))
    return m, sd, (c ** 3).mean(axis) / sd ** 3, (c ** 4).mean(axis) / sd ** 4


def corr(x, y):
    xc = x - x.mean(-1, keepdims=True); yc = y - y.mean(-1, keepdims=True)
    return (xc * yc).sum(-1) / np.sqrt((xc ** 2).sum(-1) * (yc ** 2).sum(-1) + 1e-30)


def draw_world(w: World):

    rng = np.random.default_rng(w.seed)
    P, N, T0 = w.paths, w.n_rows, w.t0
    real = rng.random((P, N)) < w.p_real
    shape = rng.choice(4, size=(P, N), p=np.array(w.mix))
    a = np.array([SHAPES[s][0] for s in SHAPE_NAMES])[shape]
    k = np.array([w.put_k if s == "put" else SHAPES[s][1] for s in SHAPE_NAMES])[shape]
    arrive = rng.integers(0, w.birth_window, (P, N))
    steps = T0 + MONTHS
    alive = np.zeros((P, N, steps), bool)
    cols = 1 if w.common_regime else N
    st = rng.random((P, cols)) < w.p_revive / (w.p_die + w.p_revive)
    for s in range(steps):
        alive[:, :, s] = st
        u = rng.random((P, cols))
        st = np.where(st, u >= w.p_die, u < w.p_revive)
    edge = np.where(real[:, :, None] & alive, w.edge, 0.0)
    core_pre = CC.draw_excess(rng, (P, T0))
    r1, r0 = math.sqrt(w.rho), math.sqrt(1 - w.rho)
    pre = (edge[:, :, :T0] + shape_ret(core_pre[:, None, :], a[:, :, None], k[:, :, None])
           + S * (r0 * rng.normal(0, 1, (P, N, T0)) + (r1 * rng.normal(0, 1, (P, 1, T0)) if w.rho else 0.0)))
    core_live = CC.draw_excess(rng, (MONTHS, P))
    noise_live = S * (r0 * rng.normal(0, 1, (MONTHS, P, N)) + (r1 * rng.normal(0, 1, (MONTHS, P, 1)) if w.rho else 0.0))
    return dict(real=real, shape=shape, a=a, k=k, arrive=arrive, edge_live=edge[:, :, T0:], core_pre=core_pre, pre=pre,
                core_live=core_live, noise_live=noise_live)


def admission(w: World, d, arm: Arm):
    pre, cp = d["pre"], d["core_pre"][:, None, :]
    P, N, T0 = pre.shape
    if arm.rule in ("all", "stake"):
        return np.ones((P, N), bool)
    if arm.rule == "oracle_noput":
        return d["shape"] != SHAPE_NAMES.index("put")
    if arm.rule == "full":
        return corr(np.broadcast_to(cp, pre.shape), pre) <= arm.thresh
    if arm.rule in ("down", "downbeta"):
        q = np.quantile(d["core_pre"], 0.10, axis=1, method="lower")
        sel = d["core_pre"] <= q[:, None]
        ntail = sel.sum(1).min()
        idx = np.argsort(np.where(sel, d["core_pre"], np.inf), axis=1)[:, :ntail]
        x = np.take_along_axis(d["core_pre"], idx, 1)[:, None, :]
        y = np.take_along_axis(pre, np.broadcast_to(idx[:, None, :], (P, N, ntail)), 2)
        if arm.rule == "down":
            return corr(np.broadcast_to(x, y.shape), y) <= arm.thresh
        xc = x - x.mean(-1, keepdims=True)
        beta = (xc * (y - y.mean(-1, keepdims=True))).sum(-1) / ((xc ** 2).sum(-1) + 1e-30)
        return beta <= arm.thresh
    m, sd, sk, ku = moments(pre)
    sr = m / sd
    if arm.rule == "psr":
        return psr(sr, T0, sk, ku) >= arm.thresh
    if arm.rule == "t3":
        return sr * math.sqrt(T0) >= arm.thresh
    if arm.rule == "dsr":
        v = sr.var(1, keepdims=True)
        n = max(N, 2)
        maxz = (1 - EULER) * ND.inv_cdf(1 - 1 / n) + EULER * ND.inv_cdf(1 - 1 / (n * math.e))
        return psr(sr, T0, sk, ku, sr0=np.sqrt(v) * maxz) >= arm.thresh
    raise ValueError(arm.rule)


def posterior(n, ysum, se1):

    p, m1, v1 = 0.5, HAIRCUT * BOOK, BOOK ** 2
    nn = np.maximum(n, 1)
    rbar, se2 = ysum / nn, se1 / nn
    l0 = np.exp(-rbar ** 2 / (2 * se2)) / np.sqrt(se2)
    l1 = np.exp(-(rbar - m1) ** 2 / (2 * (se2 + v1))) / np.sqrt(se2 + v1)
    q = np.where(n > 0, p * l1 / (p * l1 + (1 - p) * l0 + 1e-300), p)
    v = np.where(n > 0, 1 / (1 / v1 + 1 / se2), v1)
    m = np.where(n > 0, v * (m1 / v1 + rbar / se2), m1)
    mean = q * m
    return mean, np.maximum(1e-12, q * (v + m * m) - mean ** 2)


def project(z, lo, hi, budget):
    x = np.clip(z, lo, hi)
    over = x.sum(1) > budget
    if not over.any():
        return x
    zo, lo_o, hi_o, bo = z[over], lo[over], hi[over], budget[over]
    a = np.zeros(zo.shape[0]); b = np.maximum(zo.max(1), 0.0) + 1e-9
    for _ in range(40):
        t = (a + b) / 2
        s = np.clip(zo - t[:, None], lo_o, hi_o).sum(1)
        big = s > bo
        a = np.where(big, t, a); b = np.where(big, b, t)
    x[over] = np.clip(zo - b[:, None], lo_o, hi_o)
    return x


def solve(mu, bvec, dvec, lo, hi, budget, kap, x0, iters, v2=0.0):
\

    L = kap * (VC * (bvec ** 2).sum(1) + v2 * (bvec.shape[1] - 1) + dvec.max(1)) + 1e-12
    step = (1 / L)[:, None]
    x = project(x0, lo, hi, budget); yv = x.copy(); tk = 1.0
    for _ in range(iters):
        f = (bvec * yv).sum(1)
        g = mu - kap[:, None] * (VC * bvec * f[:, None] + dvec * yv)
        if v2:
            g[:, 1:] -= kap[:, None] * v2 * yv[:, 1:].sum(1, keepdims=True)
        xn = project(yv + step * g, lo, hi, budget)
        tn = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
        yv = xn + ((tk - 1) / tn) * (xn - x)
        x, tk = xn, tn
    return x


def run(w: World, arm: Arm, d=None, iters=25, stake=0.005, stake_months=24, promote=0.95):
    d = d if d is not None else draw_world(w)
    P, N = d["real"].shape
    adm = admission(w, d, arm)
    if arm.drop_fakes:
        adm = adm & d["real"]
    if arm.core_only:
        adm = np.zeros_like(adm)
    pre, cp = d["pre"], d["core_pre"]
    cpc = (cp - EX_C)[:, None, :]
    Sxy = (cpc * pre).sum(2); Sxx = np.broadcast_to((cpc ** 2).sum(2), (P, N)).copy()
    Sx = np.broadcast_to(cpc.sum(2), (P, N)).copy(); Sy = pre.sum(2); nb = np.full((P, N), float(pre.shape[2]))
    ysum = np.zeros((P, N)); y2 = np.zeros((P, N)); y3 = np.zeros((P, N)); y4 = np.zeros((P, N)); nlive = np.zeros((P, N))
    promoted = np.zeros((P, N), bool)
    W = np.ones(P); x = np.zeros((P, N + 1)); sat_share = 0.0; fake_share = 0.0
    n_admit = adm.sum(1)
    for t in range(MONTHS):
        rem = max(0, 12 * HC_YEARS - t)
        HC = sum(Y / 12 * (1 + RF) ** -j for j in range(1, rem + 1)) if rem else 0.0
        kap = G / (W + HC)
        live = adm & (d["arrive"] <= t)
        bhat = (Sxy - Sx * Sy / nb) / np.maximum(Sxx - Sx * Sx / nb, 1e-12)
        mean, var = posterior(nlive, ysum, S * S + bhat ** 2 * VC)
        if arm.rule == "stake":
            n1 = np.maximum(nlive, 2)
            m = ysum / n1; v = np.maximum(y2 / n1 - m * m, 1e-12); sd = np.sqrt(v)
            sk = (y3 / n1 - 3 * m * y2 / n1 + 2 * m ** 3) / sd ** 3
            ku = (y4 / n1 - 4 * m * y3 / n1 + 6 * m * m * y2 / n1 - 3 * m ** 4) / v ** 2
            promoted |= live & (nlive >= stake_months) & (psr(m / sd, n1, sk, ku) >= promote)
            fixed = np.where(live & ~promoted, stake * W[:, None], 0.0)
            sized = live & promoted
        else:
            fixed = np.zeros((P, N)); sized = live
        mu = np.concatenate([np.full((P, 1), EX_C), mean], 1)
        bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
        v2 = w.rho * S * S if w.sizer_rho else 0.0
        dvec = np.concatenate([np.zeros((P, 1)), S * S * (1 - (w.rho if w.sizer_rho else 0.0)) + var], 1)
        hi = np.concatenate([W[:, None], np.where(sized, CAP * W[:, None], 0.0)], 1)
        budget = W - fixed.sum(1)
        mu_adj = mu - kap[:, None] * VC * bvec * (bhat * fixed).sum(1)[:, None]
        x0 = np.clip(x * (W / np.maximum(x.sum(1), 1e-12))[:, None] if t else np.c_[W, np.zeros((P, N))], 0, hi)
        x = solve(mu_adj, bvec, dvec, np.zeros_like(hi), hi, budget, kap, x0, iters if t else 4 * iters, v2)
        hold = x[:, 1:] + fixed
        rc = d["core_live"][t]
        ry = d["edge_live"][:, :, t] + shape_ret(rc[:, None], d["a"], d["k"]) + d["noise_live"][t]
        cash = W - x[:, 0] - hold.sum(1)
        sat_share += (hold.sum(1) / W).mean() / MONTHS; fake_share += ((hold * ~d["real"]).sum(1) / W).mean() / MONTHS
        W = np.maximum(x[:, 0] * (1 + RF + rc) + (hold * (1 + RF + ry)).sum(1) + cash * (1 + RF) + Y / 12, 1e-9)
        ob = live.astype(float)
        nlive += ob; ysum += ob * ry; y2 += ob * ry ** 2; y3 += ob * ry ** 3; y4 += ob * ry ** 4
        rcc = (rc - EX_C)[:, None]
        Sxy += ob * rcc * ry; Sxx += ob * rcc ** 2; Sx += ob * rcc; Sy += ob * ry; nb += ob
    info = dict(admit_real=(adm & d["real"]).sum(1).mean(), admit_fake=(adm & ~d["real"]).sum(1).mean(),
                admit_by_shape=[(adm & (d["shape"] == s)).sum() / max((d["shape"] == s).sum(), 1) for s in range(4)],
                promoted=promoted.sum(1).mean(), promoted_real=(promoted & d["real"]).sum(1).mean(),
                n_admit=n_admit.mean(), sat_share=sat_share, fake_share=fake_share)
    return W, info


def cer(W):
    return math.log(1 / np.mean(1 / W)) / YEARS


def boot_diff(Wa, Wb, reps=400, seed=3):
    rng = np.random.default_rng(seed)
    P = Wa.shape[0]
    out = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, P, P)
        out[r] = cer(Wa[i]) - cer(Wb[i])
    return cer(Wa) - cer(Wb), out.std()
