
\
\
\
\
\
\
\
\
\

import math
from dataclasses import dataclass, field
import numpy as np
import b2_engine as E
import factory_engine as F
import core_calib as CC
import lifecycle_engine as L

FAM = 4
RHO = 0.5
TAU = 0.5
Z95 = 1.6448536269514722
P_BASE = 1 - F.P0


@dataclass(frozen=True)
class LWorld2:
    lw: L.LWorld = field(default_factory=L.LWorld)
    fam: str | None = None


@dataclass(frozen=True)
class LArm2(L.LArm):
    cap_sum: float | None = None
    retire: str = "floor"


ARMS = {
    "core": LArm2("core", core_only=True),
    "A": LArm2("A"),
    "E": LArm2("E", "family", cap_sum=0.07),
    "F": LArm2("F", retire="mono"),
    "G": LArm2("G", "family", cap_sum=0.07, retire="mono"),
    "E_nc": LArm2("E_nc", "family"),
    "E_cap0": LArm2("E_cap0", "family", cap_sum=0.0),
}


def draw2(lw2: LWorld2):
    d = L.draw(lw2.lw) if lw2.fam is None else _draw_family(lw2.lw, lw2.fam)
    d["fam"] = np.arange(d["real"].shape[1]) // FAM
    return d


def _draw_family(lw: L.LWorld, mode: str):
\

    assert lw.kind == "monthly" and lw.half_life is None
    w = lw.w
    rng = np.random.default_rng(w.seed)
    P, N, T0, M = w.paths, w.n_rows, w.t0, E.MONTHS
    assert N % FAM == 0
    rng.random((P, N))
    shape = rng.choice(4, size=(P, N), p=np.array(w.mix))
    a = np.array([E.SHAPES[s][0] for s in E.SHAPE_NAMES])[shape]
    k = np.array([w.put_k if s == "put" else E.SHAPES[s][1] for s in E.SHAPE_NAMES])[shape]
    arrive = rng.integers(0, w.birth_window, (P, N))
    steps = T0 + M
    alive = np.zeros((P, N, steps), bool)
    st = rng.random((P, N)) < w.p_revive / (w.p_die + w.p_revive)
    for s in range(steps):
        alive[:, :, s] = st
        u = rng.random((P, N))
        st = np.where(st, u >= w.p_die, u < w.p_revive)
    core_pre = CC.draw_excess(rng, (P, T0))
    noise_pre = rng.normal(0, 1, (P, N, T0))
    core_live = CC.draw_excess(rng, (M, P))
    noise_live = rng.normal(0, 1, (M, P, N))
    rng2 = np.random.default_rng(w.seed + 10_000)
    has_replay = rng2.random((P, N)) < lw.replay_frac
    rng3 = np.random.default_rng(w.seed + 20_000)
    fam_real = np.repeat(rng3.random((P, N // FAM)) < w.p_real, FAM, axis=1)
    if mode == "shared":
        real = fam_real
        e_row = w.edge * np.maximum(0.0, 1 + TAU * rng3.normal(0, 1, (P, N)))
    elif mode == "false":
        real = np.where(has_replay, fam_real, rng3.random((P, N)) < w.p_real)
        e_row = np.full((P, N), w.edge)
    else:
        raise ValueError(mode)
    ra = real[:, :, None] & alive
    edge = np.where(ra, e_row[:, :, None], 0.0)
    sd = E.S
    pre = edge[:, :, :T0] + E.shape_ret(core_pre[:, None, :], a[:, :, None], k[:, :, None]) + sd * noise_pre
    return dict(real=real, shape=shape, a=a, k=k, arrive=arrive, core_pre=core_pre, core_live=core_live,
                has_replay=has_replay, alive_live=ra[:, :, T0:], pre=np.maximum(pre, -1.0), edge_live=edge[:, :, T0:],
                noise_live=sd * noise_live, sd_obs=sd, floor=-1.0)


def _logit(q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return np.log(q) - np.log1p(-q)


def _sib(x, fam_n):

    P, N = x.shape
    tot = x.reshape(P, fam_n, FAM).sum(2)
    return np.repeat(tot, FAM, axis=1) - x


def family_prior(q_o, m_o, v_o, ev, so):
\
\
\

    P, N = q_o.shape
    fn = N // FAM
    evf = ev.astype(float)
    n_sib = _sib(evf, fn)
    llr = _sib(np.where(ev, _logit(q_o) - _logit(P_BASE), 0.0), fn)
    q_fam = 1 / (1 + np.exp(-(_logit(P_BASE) + llr)))
    wq = np.where(ev, q_o, 0.0)
    sq, sqm, sqv = _sib(wq, fn), _sib(wq * m_o, fn), _sib(wq * v_o, fn)
    mb = L.BASE_SR * so
    vb = mb * mb
    okq = sq > 1e-12
    m_fam = np.where(okq, sqm / np.maximum(sq, 1e-12), mb)
    v_fam = np.where(okq, sqv / np.maximum(sq, 1e-12), vb) + (TAU * m_fam) ** 2
    w1 = RHO * q_fam
    w2 = (1 - RHO) * P_BASE
    pE = w1 + w2
    mE = (w1 * m_fam + w2 * mb) / pE
    vE = (w1 * (v_fam + m_fam ** 2) + w2 * (vb + mb ** 2)) / pE - mE ** 2
    none = n_sib < 0.5
    return (np.where(none, P_BASE, pE), np.where(none, mb, mE), np.maximum(np.where(none, vb, vE), 1e-10))


def mono_retire(m, v):
    return m + Z95 * np.sqrt(v) <= 0


def run2(lw2: LWorld2, arm: LArm2, d=None, iters=25):
    lw = lw2.lw
    d = d if d is not None else draw2(lw2)
    P, N = d["real"].shape
    rare = lw.kind == "rare"
    fam_arm = arm.no_replay == "family"
    p, m1, v1, drop = L.priors(lw, d, arm)
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
    W = np.ones(P); x = np.zeros((P, N + 1)); sat = 0.0; fake = 0.0; nrs = 0.0
    dead_by = np.zeros((P, N), bool)
    for t in range(E.MONTHS):
        rem = max(0, 12 * E.HC_YEARS - t)
        HC = sum(E.Y / 12 * (1 + E.RF) ** -j for j in range(1, rem + 1)) if rem else 0.0
        kap = E.G / (W + HC)
        live = admitted & (d["arrive"] <= t)
        if rare:
            bhat = np.zeros((P, N))
            lo_c = np.maximum(d["arrive"], t - L.HOLD + 1)
            open_mask = (np.arange(E.MONTHS)[None, None, :] >= lo_c[:, :, None]) & (np.arange(E.MONTHS)[None, None, :] <= t)
            n_open = (d["c"] * open_mask).sum(2)
            s_open = (d["s1"] * open_mask).sum(2)
        else:
            ok = nb >= 6
            bhat = np.where(ok, (Sxy - Sx * Sy / np.maximum(nb, 1)) / np.maximum(Sxx - Sx * Sx / np.maximum(nb, 1), 1e-12), 0.0)
        se1 = so2 + bhat ** 2 * E.VC
        pt, m1t, v1t = p, m1, v1
        if fam_arm:
            q_o, m_o, v_o = F.slab_parts(nlive, ysum, se1, p, m1, v1)
            ev = (d["arrive"] <= t) & (hr | (nlive > 0))
            pf, mf, vf = family_prior(q_o, m_o, v_o, ev, d["sd_obs"])
            pt, m1t, v1t = np.where(hr, p, pf), np.where(hr, m1, mf), np.where(hr, v1, vf)
        mean, var, q = F.spike_slab(nlive, ysum, se1, pt, m1t, v1t)
        if arm.retire == "mono":
            _, ms, vs = F.slab_parts(nlive, ysum, se1, pt, m1t, v1t)
            dn = live & ~suspended & (nlive >= 1) & mono_retire(ms, vs)
            dead_by |= dn; suspended |= dn
        elif arm.q_dead is not None:
            dn = live & ~suspended & (nlive >= arm.n_dead) & (q < arm.q_dead)
            dead_by |= dn; suspended |= dn
        sized = live & ~suspended & (nlive >= arm.shadow_k)
        risk = so2 / np.maximum(n_open, 1) if rare else so2
        if rare:
            sized &= n_open > 0
        row_cap = arm.cap
        if arm.cap_sum is not None:
            k_nr = (sized & ~hr).sum(1, keepdims=True)
            row_cap = np.where(hr, arm.cap, np.minimum(arm.cap, arm.cap_sum / np.maximum(k_nr, 1)))
        mu = np.concatenate([np.full((P, 1), E.EX_C), mean], 1)
        bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
        dvec = np.concatenate([np.zeros((P, 1)), risk + var], 1)
        hi = np.concatenate([W[:, None], np.where(sized, row_cap * W[:, None], 0.0)], 1)
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
        nrs += ((hold * ~hr).sum(1) / W).mean() / E.MONTHS
        W = np.maximum(x[:, 0] * (1 + E.RF + rc) + (hold * (1 + E.RF + ry)).sum(1) + cash * (1 + E.RF) + E.Y / 12, 1e-9)
        if rare:
            s = t - L.HOLD + 1
            if s >= 0:
                obs = live & (d["arrive"] <= s)
                nlive += np.where(obs, d["c"][:, :, s], 0.0); ysum += np.where(obs, d["s1"][:, :, s], 0.0)
        else:
            ob = live.astype(float)
            nlive += ob; ysum += ob * ry
            rcc = (rc - E.EX_C)[:, None]
            Sxy += ob * rcc * ry; Sxx = Sxx + ob * rcc ** 2; Sx = Sx + ob * rcc; Sy += ob * ry; nb = nb + ob
    real = d["real"]
    info = dict(sat_share=sat, fake_share=fake, noreplay_share=nrs,
                retired_real=(dead_by & real).sum() / max(real.sum(), 1),
                retired_fake=(dead_by & ~real).sum() / max((~real).sum(), 1),
                dropped_replay=drop.sum() / max(hr.sum(), 1))
    return W, info
