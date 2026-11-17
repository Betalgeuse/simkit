
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

P0, LAM_SR, M_KR = 0.444, 0.555 / (15 / math.sqrt(12)), 0.548 / 0.742
_erfc = np.frompyfunc(math.erfc, 1, 1)


def ndtr(x):
    return 0.5 * np.asarray(_erfc(-np.asarray(x, float) / math.sqrt(2)), float)


def phi(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def replay_prior(t, n, K=1, p0=P0, mean_sr=LAM_SR * M_KR):

    t = np.asarray(t, float); a = math.sqrt(n)
    c = t / a - 1 / (mean_sr * a * a)
    ac = a * c
    Phi_ac = np.maximum(ndtr(ac), 1e-300)
    l1 = np.exp(0.5 * ac * ac - 0.5 * t * t) * Phi_ac / (mean_sr * a)
    l0 = K * phi(t) * ndtr(t) ** (K - 1)
    p = (1 - p0) * l1 / ((1 - p0) * l1 + p0 * l0)
    lam = phi(ac) / Phi_ac
    es = c + lam / a
    vs = np.maximum((1 - ac * lam - lam * lam) / (a * a), 1e-12)
    return p, es, vs


def cusum_h(k, arl0):

    k = np.asarray(k, float)
    lo, hi = np.zeros_like(k), np.full_like(k, 1e4)
    for _ in range(200):
        h = (lo + hi) / 2
        x = 2 * k * (h + 1.166)
        arl = (np.expm1(np.minimum(x, 700)) - x) / (2 * k * k)
        big = arl > arl0
        hi = np.where(big, h, hi); lo = np.where(big, lo, h)
    return (lo + hi) / 2


def siegmund_arl(drift, h):

    b = h + 1.166
    d = np.where(np.abs(drift) < 1e-9, 1e-9, drift)
    return (np.exp(-2 * d * b) + 2 * d * b - 1) / (2 * d * d)


def exp_max_z(K):

    if K <= 1:
        return 0.0
    return (1 - E.EULER) * E.ND.inv_cdf(1 - 1 / K) + E.EULER * E.ND.inv_cdf(1 - 1 / (K * math.e))


@dataclass(frozen=True)
class Factory:
    name: str = "f_cusum"
    prior: str = "replay"
    monitor: str = "cusum"
    arl0: float = 200.0
    dsr_floor: float | None = 0.5
    n_review: int = 40
    q_dead: float | None = 0.05
    K: int = 1
    hazard: float = 1 / 36


def slab_parts(n, ysum, se1, p, m1, v1):

    nn = np.maximum(n, 1)
    rbar, se2 = ysum / nn, se1 / nn
    l0 = np.exp(-rbar ** 2 / (2 * se2)) / np.sqrt(se2)
    l1 = np.exp(-(rbar - m1) ** 2 / (2 * (se2 + v1))) / np.sqrt(se2 + v1)
    q = np.where(n > 0, p * l1 / (p * l1 + (1 - p) * l0 + 1e-300), p)
    v = np.where(n > 0, 1 / (1 / v1 + 1 / se2), v1)
    m = np.where(n > 0, v * (m1 / v1 + rbar / se2), m1)
    return q, m, v


def spike_slab(n, ysum, se1, p, m1, v1):

    q, m, v = slab_parts(n, ysum, se1, p, m1, v1)
    mean = q * m
    return mean, np.maximum(1e-12, q * (v + m * m) - mean ** 2), q


def row_priors(d, f: Factory):

    pre = d.get("pre_sel", d["pre"])
    P, N, T0 = pre.shape
    if f.prior == "a0":
        p = np.full((P, N), 0.5); m1 = np.full((P, N), E.HAIRCUT * E.BOOK); v1 = np.full((P, N), E.BOOK ** 2)
        s = pre.std(2)
        return p, m1, v1, m1.copy(), s
    m, sd = pre.mean(2), pre.std(2, ddof=1)
    t = m / sd * math.sqrt(T0)
    q, es, vs = replay_prior(t, T0, f.K)
    m1 = es * sd
    return q, m1, np.maximum(vs * sd * sd, 1e-10), m1.copy(), sd


def run_factory(w: E.World, f: Factory, d=None, iters=25, family_one=None, trace=None):
    d = d if d is not None else E.draw_world(w)
    P, N = d["real"].shape
    family_one = w.common_regime if family_one is None else family_one
    pre, cp = d.get("pre_sel", d["pre"]), d["core_pre"]
    pr, m1, v1, m_alive, s_pre = row_priors(d, f)
    k = 0.5 * m_alive / s_pre
    h = cusum_h(k, f.arl0) if f.monitor == "cusum" else None
    cpc = (cp - E.EX_C)[:, None, :]
    Sxy = (cpc * pre).sum(2); Sxx = np.broadcast_to((cpc ** 2).sum(2), (P, N)).copy()
    Sx = np.broadcast_to(cpc.sum(2), (P, N)).copy(); Sy = pre.sum(2); nb = np.full((P, N), float(pre.shape[2]))
    ysum = np.zeros((P, N)); y2 = np.zeros((P, N)); y3 = np.zeros((P, N)); y4 = np.zeros((P, N)); nlive = np.zeros((P, N))
    C = np.zeros((P, N)); shrink = np.zeros((P, N), bool); nrec = np.zeros((P, N)); yrec = np.zeros((P, N))
    alarms = np.zeros((P, N)); alarm_alive = np.zeros((P, N)); first_alarm = np.full((P, N), -1)
    suspended = np.zeros((P, N), bool); why_dsr = np.zeros((P, N), bool); why_dead = np.zeros((P, N), bool)
    sr0 = exp_max_z(f.K) * math.sqrt(1 / f.n_review)
    if f.monitor == "bocpd":
        R = E.MONTHS + 1
        rl = np.zeros((P, N, R)); rl[:, :, 0] = 1.0
        rsum = np.zeros((P, N, R)); rn = np.arange(R, dtype=float)[None, None, :]
    W = np.ones(P); x = np.zeros((P, N + 1)); sat_share = 0.0; fake_share = 0.0
    alive_live = d.get("alive_live", d["edge_live"] > 0)
    for t in range(E.MONTHS):
        rem = max(0, 12 * E.HC_YEARS - t)
        HC = sum(E.Y / 12 * (1 + E.RF) ** -j for j in range(1, rem + 1)) if rem else 0.0
        kap = E.G / (W + HC)
        live = d["arrive"] <= t
        bhat = (Sxy - Sx * Sy / nb) / np.maximum(Sxx - Sx * Sx / nb, 1e-12)
        se1 = E.S * E.S + bhat ** 2 * E.VC
        mean, var, q = spike_slab(nlive, ysum, se1, pr, m1, v1)
        if f.monitor == "cusum":
            rmean, rvar, _ = spike_slab(nrec, yrec, se1, pr, m1, v1)
            use_r = shrink & (rmean / (se1 + rvar) < mean / (se1 + var))
            mean = np.where(use_r, rmean, mean); var = np.where(use_r, rvar, var)
        elif f.monitor == "bocpd":
            L = t + 1
            mr, vr, _ = spike_slab(rn[:, :, :L], rsum[:, :, :L], se1[:, :, None], pr[:, :, None], m1[:, :, None], v1[:, :, None])
            bm = (rl[:, :, :L] * mr).sum(2)
            bv = np.maximum((rl[:, :, :L] * (vr + mr * mr)).sum(2) - bm * bm, 1e-12)
            del mr, vr
            mean = np.where(live & (nlive > 0), bm, mean); var = np.where(live & (nlive > 0), bv, var)
        if f.q_dead is not None:
            dead_now = live & ~suspended & (nlive >= f.n_review) & (q < f.q_dead)
            why_dead |= dead_now; suspended |= dead_now
        sized = live & ~suspended
        mu = np.concatenate([np.full((P, 1), E.EX_C), mean], 1)
        bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
        v2 = w.rho * E.S * E.S if w.sizer_rho else 0.0
        dvec = np.concatenate([np.zeros((P, 1)), E.S * E.S * (1 - (w.rho if w.sizer_rho else 0.0)) + var], 1)
        hi = np.concatenate([W[:, None], np.where(sized, E.CAP * W[:, None], 0.0)], 1)
        x0 = np.clip(x * (W / np.maximum(x.sum(1), 1e-12))[:, None] if t else np.c_[W, np.zeros((P, N))], 0, hi)
        x = E.solve(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, x0, iters if t else 4 * iters, v2)
        hold = x[:, 1:]
        if trace is not None:
            trace.append(((hold / W[:, None]).copy(), W.copy(), kap.copy()))
        rc = d["core_live"][t]
        ry = d["edge_live"][:, :, t] + E.shape_ret(rc[:, None], d["a"], d["k"]) + d["noise_live"][t]
        cash = W - x[:, 0] - hold.sum(1)
        sat_share += (hold.sum(1) / W).mean() / E.MONTHS; fake_share += ((hold * ~d["real"]).sum(1) / W).mean() / E.MONTHS
        W = np.maximum(x[:, 0] * (1 + E.RF + rc) + (hold * (1 + E.RF + ry)).sum(1) + cash * (1 + E.RF) + E.Y / 12, 1e-9)
        ob = live.astype(float)
        nlive += ob; ysum += ob * ry; y2 += ob * ry ** 2; y3 += ob * ry ** 3; y4 += ob * ry ** 4
        if f.monitor == "cusum":
            nrec += ob; yrec += ob * ry
            z = (ry - m_alive) / s_pre
            C = np.where(live, np.maximum(0.0, C - z - k), C)
            al = live & (C > h)
            C = np.where(al, 0.0, C)
            alarms += al
            if trace is not None:
                trace.append(('alarm', t, al.copy()))
            if alive_live is not None:
                alarm_alive += al & alive_live[:, :, t]
            first_alarm = np.where(al & (first_alarm < 0), t, first_alarm)
            restart = (al.any(1, keepdims=True) & np.ones((P, N), bool)) if family_one else al
            shrink |= restart
            nrec = np.where(restart, 0.0, nrec); yrec = np.where(restart, 0.0, yrec)
        elif f.monitor == "bocpd":
            L = t + 1
            qr, mc, vc = slab_parts(rn[:, :, :L], rsum[:, :, :L], se1[:, :, None], pr[:, :, None], m1[:, :, None], v1[:, :, None])
            se = se1[:, :, None]; y = ry[:, :, None]
            pred = (qr * np.exp(-(y - mc) ** 2 / (2 * (se + vc))) / np.sqrt(se + vc)
                    + (1 - qr) * np.exp(-y * y / (2 * se)) / np.sqrt(se))
            del qr, mc, vc
            gp = rl[:, :, :L] * pred
            new = np.zeros((P, N, L + 1))
            new[:, :, 1:] = gp * (1 - f.hazard)
            new[:, :, 0] = gp.sum(2) * f.hazard
            new /= np.maximum(new.sum(2, keepdims=True), 1e-300)
            ns = np.zeros((P, N, L + 1)); ns[:, :, 1:] = rsum[:, :, :L] + y
            lv = live[:, :, None]
            rl[:, :, :L + 1] = np.where(lv, new, rl[:, :, :L + 1]); rsum[:, :, :L + 1] = np.where(lv, ns, rsum[:, :, :L + 1])
            del gp, new, ns, pred
        if f.dsr_floor is not None:
            look = live & ~suspended & (nlive == f.n_review)
            if look.any():
                n1 = float(f.n_review)
                m = ysum / n1; v = np.maximum(y2 / n1 - m * m, 1e-12); sd = np.sqrt(v)
                sk = (y3 / n1 - 3 * m * y2 / n1 + 2 * m ** 3) / sd ** 3
                ku = (y4 / n1 - 4 * m * y3 / n1 + 6 * m * m * y2 / n1 - 3 * m ** 4) / v ** 2
                fail = look & (E.psr(m / sd, n1, sk, ku, sr0=sr0) < f.dsr_floor)
                why_dsr |= fail; suspended |= fail
        rcc = (rc - E.EX_C)[:, None]
        Sxy += ob * rcc * ry; Sxx += ob * rcc ** 2; Sx += ob * rcc; Sy += ob * ry; nb += ob
    real = d["real"]
    live_n = np.maximum(nlive, 1)
    info = dict(sat_share=sat_share, fake_share=fake_share,
                alarms_real=alarms[real].sum() / max(nlive[real].sum(), 1) * 200,
                alarms_fake=alarms[~real].sum() / max(nlive[~real].sum(), 1) * 200,
                alarm_alive_real=alarm_alive[real].sum() / max(nlive[real].sum(), 1) * 200,
                susp_real=(suspended & real).sum() / max(real.sum(), 1), susp_fake=(suspended & ~real).sum() / max((~real).sum(), 1),
                dsr_real=(why_dsr & real).sum() / max(real.sum(), 1), dsr_fake=(why_dsr & ~real).sum() / max((~real).sum(), 1),
                dead_real=(why_dead & real).sum() / max(real.sum(), 1), dead_fake=(why_dead & ~real).sum() / max((~real).sum(), 1),
                prior_q_real=pr[real].mean(), prior_q_fake=pr[~real].mean(), k_med=float(np.median(k)),
                h_med=float(np.median(h)) if h is not None else float("nan"),
                first_alarm=first_alarm, k=k, h=h, s_pre=s_pre, live_n=live_n)
    return W, info
