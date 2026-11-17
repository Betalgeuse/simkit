
\
\
\
\
\
\
\
\

import csv, math, os
from dataclasses import dataclass, field
import numpy as np
import b2_engine as E
import factory_engine as F
import core_calib as CC

import simkit_params as SP
EVENT_NET = SP.event_pool()
EVENT_NOISE = EVENT_NET - EVENT_NET.mean()
EVENT_SD = float(EVENT_NET.std())
EVENT_EDGE = float(EVENT_NET.mean())
BASE_SR = F.LAM_SR * F.M_KR
RATE, HOLD = 7 / 12, 12
MAXC = 6


@dataclass(frozen=True)
class LWorld:
    w: E.World = field(default_factory=E.World)
    kind: str = "monthly"
    replay_frac: float = 0.5
    half_life: float | None = None


@dataclass(frozen=True)
class LArm:
    name: str
    no_replay: str = "reject"
    shadow_k: int = 0
    cap: float = E.CAP
    core_only: bool = False
    q_dead: float | None = 0.05
    n_dead: int = 40
    drop_neg_replay: bool = True


ARMS = {"core": LArm("core", core_only=True), "A": LArm("A"), "B": LArm("B", "base"),
        "C": LArm("C", "base", shadow_k=3), "D": LArm("D", "base", cap=0.0048)}


def draw(lw: LWorld):

    w = lw.w
    rng = np.random.default_rng(w.seed)
    P, N, T0, M = w.paths, w.n_rows, w.t0, E.MONTHS
    real = rng.random((P, N)) < w.p_real
    shape = rng.choice(4, size=(P, N), p=np.array(w.mix))
    a = np.array([E.SHAPES[s][0] for s in E.SHAPE_NAMES])[shape]
    k = np.array([w.put_k if s == "put" else E.SHAPES[s][1] for s in E.SHAPE_NAMES])[shape]
    arrive = rng.integers(0, w.birth_window, (P, N))
    steps = T0 + M
    alive = np.zeros((P, N, steps), bool)
    cols = 1 if w.common_regime else N
    st = rng.random((P, cols)) < w.p_revive / (w.p_die + w.p_revive)
    for s in range(steps):
        alive[:, :, s] = st
        u = rng.random((P, cols))
        st = np.where(st, u >= w.p_die, u < w.p_revive)
    ra = real[:, :, None] & alive
    edge = np.where(ra, w.edge, 0.0)
    core_pre = CC.draw_excess(rng, (P, T0))
    noise_pre = rng.normal(0, 1, (P, N, T0))
    core_live = CC.draw_excess(rng, (M, P))
    noise_live = rng.normal(0, 1, (M, P, N))
    assert w.rho == 0
    rng2 = np.random.default_rng(w.seed + 10_000)
    has_replay = rng2.random((P, N)) < lw.replay_frac
    edge_live = edge[:, :, T0:]
    if lw.half_life:
        age = np.arange(M)[None, None, :] - arrive[:, :, None]
        edge_live = edge_live * 2.0 ** (-np.maximum(age, 0) / lw.half_life)
    d = dict(real=real, shape=shape, a=a, k=k, arrive=arrive, core_pre=core_pre, core_live=core_live,
             has_replay=has_replay, alive_live=ra[:, :, T0:])
    if lw.kind == "rare":
        return _rare(d, ra, w, rng2, P, N, T0, M)
    sd = E.S
    if lw.kind == "fat":
        sd = EVENT_SD
        noise_pre = EVENT_NOISE[rng2.integers(0, EVENT_NOISE.size, noise_pre.shape)] / sd
        noise_live = EVENT_NOISE[rng2.integers(0, EVENT_NOISE.size, noise_live.shape)] / sd
    pre = edge[:, :, :T0] + E.shape_ret(core_pre[:, None, :], a[:, :, None], k[:, :, None]) + sd * noise_pre
    d.update(pre=np.maximum(pre, -1.0), edge_live=edge_live, noise_live=sd * noise_live, sd_obs=sd, floor=-1.0)
    return d


def _rare(d, ra, w, rng, P, N, T0, M):
\

    steps = T0 + M
    c = np.minimum(rng.poisson(RATE, (P, N, steps)), MAXC)
    s1 = np.zeros((P, N, steps)); s2 = np.zeros((P, N, steps))
    for j in range(MAXC):
        r = EVENT_NOISE[rng.integers(0, EVENT_NOISE.size, (P, N, steps))] + np.where(ra, EVENT_EDGE, 0.0)
        r = np.where(j < c, np.maximum(r, -1.0) / HOLD, 0.0)
        s1 += r; s2 += r * r
    pre_n = c[:, :, :T0 - HOLD].sum(2).astype(float)
    pre_s1 = s1[:, :, :T0 - HOLD].sum(2); pre_s2 = s2[:, :, :T0 - HOLD].sum(2)
    d.update(c=c[:, :, T0:].astype(float), s1=s1[:, :, T0:], pre_n=pre_n, pre_s1=pre_s1, pre_s2=pre_s2,
             sd_obs=EVENT_SD / HOLD)
    return d


def _replay_prior_rows(t, n):

    p = np.zeros_like(t); es = np.zeros_like(t); vs = np.zeros_like(t)
    for nv in np.unique(n):
        m = n == nv
        if nv < 3:
            p[m], es[m], vs[m] = 1 - F.P0, BASE_SR, BASE_SR ** 2
            continue
        p[m], es[m], vs[m] = F.replay_prior(t[m], int(nv))
    return p, es, vs


def priors(lw, d, arm):

    hr, so = d["has_replay"], d["sd_obs"]
    if lw.kind == "rare":
        n = d["pre_n"]; mean = d["pre_s1"] / np.maximum(n, 1)
        sd = np.sqrt(np.maximum(d["pre_s2"] - n * mean * mean, 1e-18) / np.maximum(n - 1, 1))
        t = mean / sd * np.sqrt(n)
        q, es, vs = _replay_prior_rows(t, n)
    else:
        pre = d["pre"]; T0 = pre.shape[2]
        mean, sd = pre.mean(2), pre.std(2, ddof=1)
        q, es, vs = F.replay_prior(mean / sd * math.sqrt(T0), T0)
    p = np.where(hr, q, 1 - F.P0)
    m1 = np.where(hr, es * sd, BASE_SR * so)
    v1 = np.where(hr, np.maximum(vs * sd * sd, 1e-10), (BASE_SR * so) ** 2)
    drop = hr & (mean <= 0) if arm.drop_neg_replay else np.zeros_like(hr)
    return p, m1, v1, drop


def run(lw: LWorld, arm: LArm, d=None, iters=25):
    d = d if d is not None else draw(lw)
    w = lw.w
    P, N = d["real"].shape
    rare = lw.kind == "rare"
    p, m1, v1, drop = priors(lw, d, arm)
    hr = d["has_replay"]
    admitted = np.zeros((P, N), bool) if arm.core_only else (hr if arm.no_replay == "reject" else np.ones((P, N), bool))
    suspended = drop.copy()
    so2 = d["sd_obs"] * d["sd_obs"]
    if rare:
        Sxy = Sxx = Sx = Sy = nb = None
    else:
        pre, cp = d["pre"], d["core_pre"]
        cpc = (cp - E.EX_C)[:, None, :]
        hrf = hr.astype(float)
        T0 = pre.shape[2]
        Sxy = (cpc * pre).sum(2) * hrf; Sxx = np.broadcast_to((cpc ** 2).sum(2), (P, N)) * hrf
        Sx = np.broadcast_to(cpc.sum(2), (P, N)) * hrf; Sy = pre.sum(2) * hrf; nb = T0 * hrf
    ysum = np.zeros((P, N)); nlive = np.zeros((P, N))
    W = np.ones(P); x = np.zeros((P, N + 1)); sat = 0.0; fake = 0.0
    dead_by = np.zeros((P, N), bool)
    for t in range(E.MONTHS):
        rem = max(0, 12 * E.HC_YEARS - t)
        HC = sum(E.Y / 12 * (1 + E.RF) ** -j for j in range(1, rem + 1)) if rem else 0.0
        kap = E.G / (W + HC)
        live = admitted & (d["arrive"] <= t)
        if rare:
            bhat = np.zeros((P, N))
            lo_c = np.maximum(d["arrive"], t - HOLD + 1)
            open_mask = (np.arange(E.MONTHS)[None, None, :] >= lo_c[:, :, None]) & (np.arange(E.MONTHS)[None, None, :] <= t)
            n_open = (d["c"] * open_mask).sum(2)
            s_open = (d["s1"] * open_mask).sum(2)
        else:
            ok = nb >= 6
            bhat = np.where(ok, (Sxy - Sx * Sy / np.maximum(nb, 1)) / np.maximum(Sxx - Sx * Sx / np.maximum(nb, 1), 1e-12), 0.0)
        se1 = so2 + bhat ** 2 * E.VC
        mean, var, q = F.spike_slab(nlive, ysum, se1, p, m1, v1)
        if arm.q_dead is not None:
            dn = live & ~suspended & (nlive >= arm.n_dead) & (q < arm.q_dead)
            dead_by |= dn; suspended |= dn
        sized = live & ~suspended & (nlive >= arm.shadow_k)
        risk = so2 / np.maximum(n_open, 1) if rare else so2
        if rare:
            sized &= n_open > 0
        mu = np.concatenate([np.full((P, 1), E.EX_C), mean], 1)
        bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
        dvec = np.concatenate([np.zeros((P, 1)), risk + var], 1)
        hi = np.concatenate([W[:, None], np.where(sized, arm.cap * W[:, None], 0.0)], 1)
        x0 = np.clip(x * (W / np.maximum(x.sum(1), 1e-12))[:, None] if t else np.c_[W, np.zeros((P, N))], 0, hi)
        x = E.solve(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, x0, iters if t else 4 * iters, 0.0)
        hold = x[:, 1:]
        rc = d["core_live"][t]
        if rare:
            ry = np.where(n_open > 0, s_open / np.maximum(n_open, 1), 0.0)
        else:
            ry = np.maximum(d["edge_live"][:, :, t] + E.shape_ret(rc[:, None], d["a"], d["k"]) + d["noise_live"][t], d["floor"])
        cash = W - x[:, 0] - hold.sum(1)
        sat += (hold.sum(1) / W).mean() / E.MONTHS; fake += ((hold * ~d["real"]).sum(1) / W).mean() / E.MONTHS
        W = np.maximum(x[:, 0] * (1 + E.RF + rc) + (hold * (1 + E.RF + ry)).sum(1) + cash * (1 + E.RF) + E.Y / 12, 1e-9)
        if rare:
            s = t - HOLD + 1
            if s >= 0:
                obs = live & (d["arrive"] <= s)
                nlive += np.where(obs, d["c"][:, :, s], 0.0); ysum += np.where(obs, d["s1"][:, :, s], 0.0)
        else:
            ob = live.astype(float)
            nlive += ob; ysum += ob * ry
            rcc = (rc - E.EX_C)[:, None]
            Sxy += ob * rcc * ry; Sxx = Sxx + ob * rcc ** 2; Sx = Sx + ob * rcc; Sy += ob * ry; nb = nb + ob
    real = d["real"]
    info = dict(sat_share=sat, fake_share=fake,
                retired_real=(dead_by & real).sum() / max(real.sum(), 1),
                retired_fake=(dead_by & ~real).sum() / max((~real).sum(), 1),
                dropped_replay=drop.sum() / max(hr.sum(), 1))
    return W, info
