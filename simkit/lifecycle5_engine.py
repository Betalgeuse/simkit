
\
\
\
\
\
\
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
from dataclasses import dataclass
import numpy as np
import b2_engine as E
import factory_engine as F
import lifecycle_engine as L
import lifecycle2_engine as L2
import lifecycle3_engine as L3
import lifecycle4_engine as L4
import jax_backend as JX

DELTA, Q_RET = L3.DELTA, L3.Q_RET
import simkit_params as SP
CUR_SLOTS, CUR_ROW_CAP, EXPLORE = SP.P['cur_slots'], SP.P['cur_row_cap'], SP.P['explore']
L4_ROW_CAP = E.CAP
RHO_FAM = L2.RHO


@dataclass(frozen=True)
class Arm5:
    name: str
    alloc: str = "L4"
    kappa: float = 1.0


def allocate(mean, var, so2, bhat, fam, sized, W, kap, kappa, x0, iters):
\
\
\
\

    P, N = mean.shape
    s = np.where(sized, np.sqrt(var), 0.0)
    nf = int(fam.max()) + 1
    onehot = (fam[:, None] == np.arange(nf)[None, :]).astype(float)
    mu = np.concatenate([np.full((P, 1), E.EX_C), kappa * mean], 1)
    bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
    dvec = np.concatenate([np.zeros((P, 1)), so2 + (1 - RHO_FAM) * var], 1)
    hi = np.concatenate([W[:, None], np.where(sized, W[:, None], 0.0)], 1)
    blk = RHO_FAM * (s * s) @ onehot
    Lc = kap * (E.VC * (bvec ** 2).sum(1) + dvec.max(1) + blk.max(1)) + 1e-12
    step = (1 / Lc)[:, None]
    if JX.ON:
        return JX.apg(mu, bvec, dvec, s, RHO_FAM, onehot, hi, W, kap, step, x0, iters, E.VC)
    lo = np.zeros_like(hi)
    x = E.project(np.clip(x0, 0, hi), lo, hi, W.copy()); yv = x.copy(); tk = 1.0
    for _ in range(iters):
        f = (bvec * yv).sum(1)
        famsum = (s * yv[:, 1:]) @ onehot
        g = mu - kap[:, None] * (E.VC * bvec * f[:, None] + dvec * yv)
        g[:, 1:] -= kap[:, None] * RHO_FAM * s * (famsum @ onehot.T)
        xn = E.project(yv + step * g, lo, hi, W.copy())
        tn = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
        yv = xn + ((tk - 1) / tn) * (xn - x)
        x, tk = xn, tn
    return x


def tz_blend(x, mean, var, so2, sized, kap):
\
\

    sat = x[:, 1:]
    n = np.maximum(sized.sum(1), 1)
    tot = sat.sum(1)
    e = np.where(sized, (tot / n)[:, None], 0.0)
    sig2 = so2 + var
    l_hat = np.where(sized, var / sig2, 0.0).sum(1) / (2 * kap)
    l_e = kap / 2 * np.where(sized, sig2 * (sat - e) ** 2, 0.0).sum(1)
    delta = np.where(l_e + l_hat > 0, l_e / np.maximum(l_e + l_hat, 1e-300), 1.0)
    out = x.copy()
    out[:, 1:] = (1 - delta)[:, None] * e + delta[:, None] * sat
    return out, delta


def cur_slots(sized, held, arrive):

    P, N = sized.shape
    key = np.where(sized, np.where(held, 0, 1) * 1e6 + arrive * 100 + np.arange(N)[None, :], np.inf)
    rank = np.argsort(np.argsort(key, axis=1), axis=1)
    return sized & (rank < CUR_SLOTS)


def cur_greedy_explore(hold, nr, order, W):

    h = np.take_along_axis(np.where(nr, hold, 0.0), order, 1)
    before = np.cumsum(h, 1) - h
    got = np.clip(EXPLORE * W[:, None] - before, 0.0, h)
    out = hold.copy()
    np.put_along_axis(out, order, np.where(np.take_along_axis(nr, order, 1), got, np.take_along_axis(hold, order, 1)), 1)
    return out


def run5(lw2: L2.LWorld2, arm: Arm5, d=None, iters=25):
    lw = lw2.lw
    d = d if d is not None else L2.draw2(lw2)
    P, N = d["real"].shape
    rare = lw.kind == "rare"
    core_only = arm.alloc == "core"
    base = L2.LArm2(arm.name, "family")
    p, m1, v1, drop = L.priors(lw, d, base)
    hr = d["has_replay"]
    fam = d["fam"]
    admitted = np.zeros((P, N), bool) if core_only else np.ones((P, N), bool)
    suspended = drop.copy()
    so2 = d["sd_obs"] * d["sd_obs"]
    order = np.argsort(np.random.default_rng(lw.w.seed + 50_000).random((P, N)), 1)
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
    W = np.ones(P); x = np.zeros((P, N + 1)); sat = 0.0; fake = 0.0; nrs = 0.0; nsz = 0.0; dsum = 0.0
    dead_by = np.zeros((P, N), bool)
    for t in range(E.MONTHS):
        rem = max(0, 12 * E.HC_YEARS - t)
        HC = sum(E.Y / 12 * (1 + E.RF) ** -j for j in range(1, rem + 1)) if rem else 0.0
        kap = E.G / (W + HC)
        live = admitted & (d["arrive"] <= t)
        if rare:
            bhat = np.zeros((P, N))
            lo_c = np.maximum(d["arrive"], t - L.HOLD + 1)
            mm = np.arange(E.MONTHS)[None, None, :]
            open_mask = (mm >= lo_c[:, :, None]) & (mm <= t)
            n_open = (d["c"] * open_mask).sum(2)
            s_open = (d["s1"] * open_mask).sum(2)
        else:
            ok = nb >= 6
            bhat = np.where(ok, (Sxy - Sx * Sy / np.maximum(nb, 1)) / np.maximum(Sxx - Sx * Sx / np.maximum(nb, 1), 1e-12), 0.0)
        se1 = so2 + bhat ** 2 * E.VC
        q_o, m_o, v_o = F.slab_parts(nlive, ysum, se1, p, m1, v1)
        ev = (d["arrive"] <= t) & (hr | (nlive > 0))
        pf, mf, vf = L2.family_prior(q_o, m_o, v_o, ev, d["sd_obs"])
        pt, m1t, v1t = np.where(hr, p, pf), np.where(hr, m1, mf), np.where(hr, v1, vf)
        mean, var, q = F.spike_slab(nlive, ysum, se1, pt, m1t, v1t)
        qs, ms, vs_ = F.slab_parts(nlive, ysum, se1, pt, m1t, v1t)
        p_le = (1 - qs) + qs * F.ndtr((DELTA - ms) / np.sqrt(vs_))
        dn = live & ~suspended & (nlive >= 1) & (p_le >= Q_RET)
        dead_by |= dn; suspended |= dn
        sized = live & ~suspended
        risk = so2 / np.maximum(n_open, 1) if rare else np.full((P, N), so2)
        if rare:
            sized &= n_open > 0
        it = iters if t else 4 * iters
        x0 = np.clip(x * (W / np.maximum(x.sum(1), 1e-12))[:, None], 0, None) if t else np.c_[W, np.zeros((P, N))]
        if core_only:
            x = np.c_[W, np.zeros((P, N))]
        elif arm.alloc in ("L4", "CUR"):
            cap = L4_ROW_CAP
            if arm.alloc == "CUR":
                sized = cur_slots(sized, x[:, 1:] > 1e-9 * W[:, None], d["arrive"])
                cap = CUR_ROW_CAP
            mu = np.concatenate([np.full((P, 1), E.EX_C), mean], 1)
            bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
            dvec = np.concatenate([np.zeros((P, 1)), risk + var], 1)
            hi = np.concatenate([W[:, None], np.where(sized, cap * W[:, None], 0.0)], 1)
            xs = np.clip(x0, 0, hi)
            if arm.alloc == "L4":
                g = np.concatenate([np.zeros((P, 1), bool), sized & ~hr], 1)
                x = L3.solve_group(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, xs, it, g, EXPLORE * W)
            else:
                x = E.solve(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, xs, it, 0.0)
                x[:, 1:] = cur_greedy_explore(x[:, 1:], sized & ~hr, order, W)
        else:
            x = allocate(mean, var, risk, bhat, fam, sized, W, kap, arm.kappa, x0, it)
            if arm.alloc == "TZ":
                x, dl = tz_blend(x, mean, var, risk, sized, kap)
                dsum += dl.mean() / E.MONTHS
        hold = x[:, 1:]
        rc = d["core_live"][t]
        if rare:
            ry = np.where(n_open > 0, s_open / np.maximum(n_open, 1), 0.0)
        else:
            ry = np.maximum(d["edge_live"][:, :, t] + E.shape_ret(rc[:, None], d["a"], d["k"]) + d["noise_live"][t], d["floor"])
        cash = W - x[:, 0] - hold.sum(1)
        sat += (hold.sum(1) / W).mean() / E.MONTHS; fake += ((hold * ~d["real"]).sum(1) / W).mean() / E.MONTHS
        nrs += ((hold * ~hr).sum(1) / W).mean() / E.MONTHS
        nsz += (hold > 1e-6 * W[:, None]).sum(1).mean() / E.MONTHS
        W = np.maximum(x[:, 0] * (1 + E.RF + rc) + (hold * (1 + E.RF + ry)).sum(1) + cash * (1 + E.RF) + E.Y / 12, 1e-9)
        if rare:
            s0 = t - L.HOLD + 1
            if s0 >= 0:
                obs = live & (d["arrive"] <= s0)
                nlive += np.where(obs, d["c"][:, :, s0], 0.0); ysum += np.where(obs, d["s1"][:, :, s0], 0.0)
        else:
            ob = live.astype(float)
            nlive += ob; ysum += ob * ry
            rcc = (rc - E.EX_C)[:, None]
            Sxy += ob * rcc * ry; Sxx = Sxx + ob * rcc ** 2; Sx = Sx + ob * rcc; Sy += ob * ry; nb = nb + ob
    real = d["real"]
    info = dict(sat_share=sat, fake_share=fake, noreplay_share=nrs, rows_held=nsz, tz_delta=dsum,
                retired_real=(dead_by & real).sum() / max(real.sum(), 1),
                retired_fake=(dead_by & ~real).sum() / max((~real).sum(), 1))
    return W, info
