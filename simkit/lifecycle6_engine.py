
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
import core_calib as CC
import factory_engine as F
import lifecycle_engine as L
import lifecycle2_engine as L2
import lifecycle3_engine as L3
import lifecycle5_engine as A5
from lifecycle4_engine import project_group_exact

DELTA, Q_RET = A5.DELTA, A5.Q_RET
BUDGET = 0.125
TAIL_Q = 0.05
L_CASE = float(-np.sort(L.EVENT_NOISE)[:max(1, int(TAIL_Q * L.EVENT_NOISE.size))].mean())


def _declared_worst():
\

    rng = np.random.default_rng(0)
    r = CC.draw_excess(rng, 2_000_000)
    out = {}
    for kind, noise in (("monthly", E.S * rng.normal(0, 1, r.size)),
                        ("fat", L.EVENT_NOISE[rng.integers(0, L.EVENT_NOISE.size, r.size)])):
        out[kind] = min(1.0, max(-np.quantile(np.maximum(E.shape_ret(r, a, k) + noise, -1.0), 0.001)
                                 for a, k in E.SHAPES.values()))
    out["rare"] = min(1.0, float(-np.quantile(np.maximum(L.EVENT_NET, -1.0), 0.001)))
    return out


DECLARED_WORST = _declared_worst()


@dataclass(frozen=True)
class Arm6:
    name: str
    alloc: str = "NEW"
    kappa: float = 1.0
    fence: str = "none"
    rare_risk: str = "base"
    tz: str = "none"


def allocate(mean, var, so2, bhat, fam, sized, W, kap, kappa, x0, iters, gb=None):

    P, N = mean.shape
    if gb is None:
        proj = lambda z, lo, hi: E.project(z, lo, hi, W.copy())
    else:
        g = np.concatenate([np.zeros((P, 1), bool), np.ones((P, N), bool)], 1)
        proj = lambda z, lo, hi: project_group_exact(z, lo, hi, W.copy(), g, gb)
    s = np.where(sized, np.sqrt(var), 0.0)
    nf = int(fam.max()) + 1
    onehot = (fam[:, None] == np.arange(nf)[None, :]).astype(float)
    mu = np.concatenate([np.full((P, 1), E.EX_C), kappa * mean], 1)
    bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
    dvec = np.concatenate([np.zeros((P, 1)), so2 + (1 - A5.RHO_FAM) * var], 1)
    hi = np.concatenate([W[:, None], np.where(sized, W[:, None], 0.0)], 1)
    blk = A5.RHO_FAM * (s * s) @ onehot
    Lc = kap * (E.VC * (bvec ** 2).sum(1) + dvec.max(1) + blk.max(1)) + 1e-12
    step = (1 / Lc)[:, None]
    lo = np.zeros_like(hi)
    x = proj(np.clip(x0, 0, hi), lo, hi); yv = x.copy(); tk = 1.0
    for _ in range(iters):
        f = (bvec * yv).sum(1)
        famsum = (s * yv[:, 1:]) @ onehot
        g_ = mu - kap[:, None] * (E.VC * bvec * f[:, None] + dvec * yv)
        g_[:, 1:] -= kap[:, None] * A5.RHO_FAM * s * (famsum @ onehot.T)
        xn = proj(yv + step * g_, lo, hi)
        tn = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
        yv = xn + ((tk - 1) / tn) * (xn - x)
        x, tk = xn, tn
    return x


def tz_delta(w, e, var, s2, sized, kap, kappa):
\
\
\

    p2 = (kappa / kap) ** 2 * np.where(sized, var / s2, 0.0).sum(1)
    p1 = np.maximum(np.where(sized, s2 * (e - w) ** 2, 0.0).sum(1) - p2, 0.0)
    return np.where(p1 + p2 > 0, p1 / np.maximum(p1 + p2, 1e-300), 1.0)


def tz_blend(x, mean, var, risk, sized, kap, kappa):

    sat = x[:, 1:]
    n = np.maximum(sized.sum(1), 1)
    e = np.where(sized, (sat.sum(1) / n)[:, None], 0.0)
    delta = tz_delta(sat, e, var, risk + var, sized, kap, kappa)
    out = x.copy()
    out[:, 1:] = (1 - delta)[:, None] * e + delta[:, None] * sat
    return out, delta


def cur_fence(hold, held, order, gb):
\

    P, N = hold.shape
    pos = np.argsort(order, 1)
    seq = np.argsort(np.where(held, 0, N) + pos, 1)
    h = np.take_along_axis(hold, seq, 1)
    before = np.cumsum(h, 1) - h
    got = np.clip(gb[:, None] - before, 0.0, h)
    out = np.empty_like(hold)
    np.put_along_axis(out, seq, got, 1)
    return out


def run6(lw2: L2.LWorld2, arm: Arm6, d=None, iters=25):
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
    worst = {"none": None, "f100": 1.0, "fw": DECLARED_WORST[lw.kind]}[arm.fence]
    tz_on = arm.tz == "all" or (arm.tz == "rare" and rare)
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
    cum = np.zeros((P, N)); hw = np.zeros((P, N))
    W = np.ones(P); x = np.zeros((P, N + 1)); sat = 0.0; fake = 0.0; nrs = 0.0; nsz = 0.0; dsum = 0.0
    bind = 0.0; room_sh = 0.0; real_sh = 0.0
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
        if rare:
            no = np.maximum(n_open, 1)
            risk = {"base": so2 / no, "lr": L.HOLD * so2 / no, "tail": L_CASE ** 2 / (L.HOLD * no)}[arm.rare_risk]
            sized &= n_open > 0
        else:
            risk = np.full((P, N), so2)
        gb = None if worst is None else np.maximum(BUDGET * W - hw.sum(1), 0.0) / worst
        it = iters if t else 4 * iters
        x0 = np.clip(x * (W / np.maximum(x.sum(1), 1e-12))[:, None], 0, None) if t else np.c_[W, np.zeros((P, N))]
        if core_only:
            x = np.c_[W, np.zeros((P, N))]
        elif arm.alloc in ("L4", "CUR"):
            cap = A5.L4_ROW_CAP
            held = x[:, 1:] > 1e-9 * W[:, None]
            if arm.alloc == "CUR":
                sized = A5.cur_slots(sized, held, d["arrive"])
                cap = A5.CUR_ROW_CAP
            mu = np.concatenate([np.full((P, 1), E.EX_C), mean], 1)
            bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
            dvec = np.concatenate([np.zeros((P, 1)), risk + var], 1)
            hi = np.concatenate([W[:, None], np.where(sized, cap * W[:, None], 0.0)], 1)
            xs = np.clip(x0, 0, hi)
            if arm.alloc == "L4":
                assert gb is None, "L4 is the c1 control only"
                g = np.concatenate([np.zeros((P, 1), bool), sized & ~hr], 1)
                x = L3.solve_group(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, xs, it, g, A5.EXPLORE * W)
            else:
                x = E.solve(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, xs, it, 0.0)
                x[:, 1:] = A5.cur_greedy_explore(x[:, 1:], sized & ~hr, order, W)
                if gb is not None:
                    x[:, 1:] = cur_fence(x[:, 1:], held, order, gb)
        else:
            x = allocate(mean, var, risk, bhat, fam, sized, W, kap, arm.kappa, x0, it, gb)
            if tz_on:
                x, dl = tz_blend(x, mean, var, risk, sized, kap, arm.kappa)
                dsum += dl.mean() / E.MONTHS
        hold = x[:, 1:]
        if gb is not None:
            bind += (hold.sum(1) >= gb - 1e-6 * W).mean() / E.MONTHS
            room_sh += (gb * worst / W).mean() / E.MONTHS
            real_sh += (hw.sum(1) / W).mean() / E.MONTHS
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
        cum += hold * ry
        hw = np.maximum(hw, -cum)
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
                fence_bind=bind, room_share=room_sh, realized_share=real_sh,
                retired_real=(dead_by & real).sum() / max(real.sum(), 1),
                retired_fake=(dead_by & ~real).sum() / max((~real).sum(), 1))
    return W, info
