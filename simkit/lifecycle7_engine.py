
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
import lifecycle5_engine as A5

S_SCEN = 64
EM_ITERS = 12
N0_BD = 60.0
CRISIS_BD = 0.5
EVENT_STD = L.EVENT_NOISE / L.EVENT_NOISE.std()
GAMMA = E.G


@dataclass(frozen=True)
class Arm7:
    name: str
    alloc: str = "L7"
    kappa: float = 1.0


def eb_fit(z, se2, ok, iters=None, m_init=0.0, tau2_init=1e-6):
\

    P, N = z.shape
    iters = EM_ITERS if iters is None else iters
    okf = ok.astype(float)
    pi1 = np.full(P, 0.5); m = np.full(P, m_init); tau2 = np.full(P, tau2_init)
    se2 = np.maximum(se2, 1e-14)
    for _ in range(iters):
        v1 = tau2[:, None] + se2
        l0 = -0.5 * z * z / se2 - 0.5 * np.log(se2)
        l1 = -0.5 * (z - m[:, None]) ** 2 / v1 - 0.5 * np.log(v1)
        a = np.log(pi1)[:, None] + l1; b = np.log1p(-pi1)[:, None] + l0
        r = okf / (1 + np.exp(np.clip(b - a, -50, 50)))
        sr = r.sum(1)
        pi1 = np.clip((sr + 1) / (okf.sum(1) + 2), 1e-3, 1 - 1e-3)
        wgt = r / v1
        m = np.where(sr > 1e-9, (wgt * z).sum(1) / np.maximum(wgt.sum(1), 1e-300), m)
        tau2 = np.maximum(np.where(sr > 1e-9, (r * ((z - m[:, None]) ** 2 - se2)).sum(1) / np.maximum(sr, 1e-300), tau2), 1e-10)
    return pi1, m, tau2


def eb_posterior_z(z, se2, ok, m_init, tau2_init):
\

    se2 = np.where(ok, np.maximum(se2, 1e-14), 1.0)
    pi1, m, tau2 = eb_fit(np.where(ok, z, 0.0), se2, ok, m_init=m_init, tau2_init=tau2_init)
    pi1, m, tau2 = pi1[:, None], m[:, None], tau2[:, None]
    l0 = np.exp(-z * z / (2 * se2)) / np.sqrt(se2)
    l1 = np.exp(-(z - m) ** 2 / (2 * (se2 + tau2))) / np.sqrt(se2 + tau2)
    q = np.where(ok, pi1 * l1 / (pi1 * l1 + (1 - pi1) * l0 + 1e-300), pi1)
    v = np.where(ok, 1 / (1 / tau2 + 1 / se2), tau2)
    mp = np.where(ok, v * (m / tau2 + z / se2), m)
    return q, mp, v, (pi1[:, 0], m[:, 0], tau2[:, 0])


def eb_posterior(n, s, se1, ok, m_init, tau2_init):
    nn = np.maximum(n, 1)
    return eb_posterior_z(s / nn, se1 / nn, ok & (n > 0), m_init, tau2_init)


def scenarios(rng, q, mp, vp, b, bd, sd, sized, fat, rare_scale, P, N):

    h = S_SCEN // 2
    anti = lambda z: np.concatenate([z, -z], 1)
    c = math.sqrt(E.VC) * anti(rng.standard_normal((P, h)))
    real = np.tile(rng.random((P, h, N)) < q[:, None, :], (1, 2, 1))
    th = np.where(real, mp[:, None, :] + np.sqrt(vp)[:, None, :] * anti(rng.standard_normal((P, h, N))), 0.0)
    eps = EVENT_STD[rng.integers(0, EVENT_STD.size, (P, S_SCEN, N))] if fat else anti(rng.standard_normal((P, h, N)))
    r = th + b[:, None, :] * c[:, :, None] + bd[:, None, :] * np.minimum(c, 0.0)[:, :, None] + (sd * rare_scale)[:, None, :] * eps
    r = np.where(sized[:, None, :], r, 0.0)
    return c + E.EX_C, r


def ce_alloc(c, r, sized, W, HC, x0, iters, Lc):
\

    P, N = sized.shape
    R = np.concatenate([c[:, :, None], r], 2)
    T = (W + HC)[:, None]
    hi = np.concatenate([W[:, None], np.where(sized, W[:, None], 0.0)], 1)
    lo = np.zeros_like(hi)
    x = E.project(np.clip(x0, 0, hi), lo, hi, W.copy()); yv = x.copy(); tk = 1.0
    step = (1 / Lc)[:, None]
    for _ in range(iters):
        g1 = np.maximum(1 + np.einsum("psk,pk->ps", R, yv) / T, 1e-3)
        g = np.einsum("ps,psk->pk", g1 ** (-GAMMA), R) / R.shape[1]
        xn = E.project(yv + step * g, lo, hi, W.copy())
        tn = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
        yv = xn + ((tk - 1) / tn) * (xn - x)
        x, tk = xn, tn
    return x


def run7(lw2: L2.LWorld2, arm: Arm7, d=None, iters=25, crisis=False):
\

    lw = lw2.lw
    d = d if d is not None else L2.draw2(lw2)
    P, N = d["real"].shape
    rare = lw.kind == "rare"
    fat = lw.kind in ("fat", "rare")
    core_only = arm.alloc == "core"
    base = L2.LArm2(arm.name, "family")
    p, m1, v1, drop = L.priors(lw, d, base)
    hr = d["has_replay"]
    fam = d["fam"]
    admitted = np.zeros((P, N), bool) if core_only else np.ones((P, N), bool)
    suspended = drop.copy()
    so2 = d["sd_obs"] * d["sd_obs"]
    order = np.argsort(np.random.default_rng(lw.w.seed + 50_000).random((P, N)), 1)
    srng = np.random.default_rng(lw.w.seed + 70_000)
    if crisis:
        cpd = np.minimum(d["core_pre"] - E.EX_C, 0.0)[:, None, :]
        d = dict(d); d["pre"] = np.maximum(d["pre"] + CRISIS_BD * cpd, -1.0)
    if rare:
        Sxy = Sxx = Sx = Sy = nb = None
        ev_n = np.where(hr, d["pre_n"], 0.0).astype(float); ev_s = np.where(hr, d["pre_s1"], 0.0)
    else:
        pre, cp = d["pre"], d["core_pre"]
        cpc = (cp - E.EX_C)[:, None, :]
        hrf = hr.astype(float)
        T0 = pre.shape[2]
        Sxy = (cpc * pre).sum(2) * hrf; Sxx = np.broadcast_to((cpc ** 2).sum(2), (P, N)) * hrf
        Sx = np.broadcast_to(cpc.sum(2), (P, N)) * hrf; Sy = pre.sum(2) * hrf; nb = T0 * hrf
        ev_n = T0 * hrf; ev_s = pre.sum(2) * hrf

        cm = np.minimum(cpc, 0.0)
        Xs = [np.ones_like(cpc), cpc, cm]
        Gm = np.zeros((P, N, 3, 3)); hv = np.zeros((P, N, 3)); Syy = (pre * pre).sum(2) * hrf
        for i in range(3):
            hv[..., i] = (Xs[i] * pre).sum(2) * hrf
            for j in range(3):
                Gm[..., i, j] = np.broadcast_to((Xs[i] * Xs[j]).sum(2), (P, N)) * hrf
    ysum = np.zeros((P, N)); nlive = np.zeros((P, N))
    W = np.ones(P); x = np.zeros((P, N + 1)); sat = 0.0; fake = 0.0; nrs = 0.0; nsz = 0.0
    cal = np.zeros(6); cal5 = np.zeros(6); hyp = np.zeros(3)
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
        if arm.alloc == "L7":
            base_m = L.BASE_SR * d["sd_obs"]
            if rare:
                q7, mp7, vp7, (pi1, mh, t2) = eb_posterior(ev_n + nlive, ev_s + ysum, se1, ev, 0.0, float(np.mean(base_m) ** 2))
                b, bd = np.zeros((P, N)), np.zeros((P, N))
                scale = np.sqrt(L.HOLD / np.maximum(n_open, 1))
            else:
                Greg = Gm + 1e-9 * np.eye(3)
                Ginv = np.linalg.inv(Greg)
                beta = (Ginv @ hv[..., None])[..., 0]
                nobs = Gm[..., 0, 0]
                okb = (nobs >= 6) & ev

                rss = np.maximum(Syy - (beta * hv).sum(-1), 0.0)
                s2r = np.where(nobs > 3, rss / np.maximum(nobs - 3, 1), so2)
                q7, mp7, vp7, (pi1, mh, t2) = eb_posterior_z(beta[..., 0], s2r * Ginv[..., 0, 0], okb, 0.0, float(np.mean(base_m) ** 2))
                bd_raw = np.where(okb, beta[..., 2], 0.0)
                wpool = okb.astype(float)
                bd_pop = (bd_raw * wpool).sum(1) / np.maximum(wpool.sum(1), 1)
                sh = nobs / (nobs + N0_BD)
                bd = np.where(okb, sh * bd_raw + (1 - sh) * bd_pop[:, None], bd_pop[:, None])
                b = np.where(okb, beta[..., 1], bhat)
                scale = np.ones((P, N))
            mean7 = q7 * mp7
            hyp += np.array([pi1.mean(), mh.mean(), np.sqrt(t2).mean()]) / E.MONTHS
            sd = np.sqrt(so2) if rare else np.sqrt(s2r)
            c_s, r_s = scenarios(srng, q7, mp7, vp7, b, bd, sd, sized, fat, scale, P, N)
            dvec = sd ** 2 * scale ** 2 + vp7 + (b ** 2 + bd ** 2) * E.VC
            Lc = GAMMA / (W + HC) * (E.VC * 4 + np.where(sized, dvec, 0.0).max(1) * 2) + 1e-12
            x = ce_alloc(c_s, r_s, sized, W, HC, x0, it, Lc)
            if not rare:
                tr = d["edge_live"][:, :, t]
                for acc, pm in ((cal, mean7), (cal5, mean)):
                    mk = live
                    xs_, ys_ = pm[mk], tr[mk]
                    acc += [xs_.size, xs_.sum(), ys_.sum(), (xs_ * ys_).sum(), (xs_ * xs_).sum(), 0.0]
        elif core_only:
            x = np.c_[W, np.zeros((P, N))]
        elif arm.alloc in ("L4", "CUR"):
            cap = A5.L4_ROW_CAP
            if arm.alloc == "CUR":
                sized = A5.cur_slots(sized, x[:, 1:] > 1e-9 * W[:, None], d["arrive"])
                cap = A5.CUR_ROW_CAP
            mu = np.concatenate([np.full((P, 1), E.EX_C), mean], 1)
            bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
            dvec = np.concatenate([np.zeros((P, 1)), risk + var], 1)
            hi = np.concatenate([W[:, None], np.where(sized, cap * W[:, None], 0.0)], 1)
            xs = np.clip(x0, 0, hi)
            if arm.alloc == "L4":
                g = np.concatenate([np.zeros((P, 1), bool), sized & ~hr], 1)
                x = L3.solve_group(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, xs, it, g, A5.EXPLORE * W)
            else:
                x = E.solve(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, xs, it, 0.0)
                x[:, 1:] = A5.cur_greedy_explore(x[:, 1:], sized & ~hr, order, W)
        else:
            x = A5.allocate(mean, var, risk, bhat, fam, sized, W, kap, arm.kappa, x0, it)
        hold = x[:, 1:]
        rc = d["core_live"][t]
        if rare:
            ry = np.where(n_open > 0, s_open / np.maximum(n_open, 1), 0.0)
        else:
            ry = np.maximum(d["edge_live"][:, :, t] + E.shape_ret(rc[:, None], d["a"], d["k"]) + d["noise_live"][t], d["floor"])
            if crisis:
                ry = np.maximum(ry + CRISIS_BD * np.minimum(rc - E.EX_C, 0.0)[:, None], d["floor"])
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
            if arm.alloc == "L7":
                xr = [np.ones_like(rcc), rcc, np.minimum(rcc, 0.0)]
                Syy += ob * ry * ry
                for i in range(3):
                    hv[..., i] += ob * xr[i] * ry
                    for j in range(3):
                        Gm[..., i, j] += ob * xr[i] * xr[j]
    real = d["real"]
    info = dict(sat_share=sat, fake_share=fake, noreplay_share=nrs, rows_held=nsz,
                retired_real=(dead_by & real).sum() / max(real.sum(), 1),
                retired_fake=(dead_by & ~real).sum() / max((~real).sum(), 1),
                eb_pi1=hyp[0], eb_m=hyp[1], eb_tau=hyp[2])
    for nm, acc in (("cal7", cal), ("cal5", cal5)):
        n_, sx, sy, sxy, sxx, _ = acc
        if n_ > 100:
            info[nm + "_slope"] = ((sxy - sx * sy / n_) / max(sxx - sx * sx / n_, 1e-300))
    return W, info


DELTA, Q_RET = A5.DELTA, A5.Q_RET
