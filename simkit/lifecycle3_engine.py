
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

import json, math, os
from dataclasses import dataclass, field
import numpy as np
import b2_engine as E
import factory_engine as F
import core_calib as CC
import lifecycle_engine as L
import lifecycle2_engine as L2

HERE = os.path.dirname(os.path.abspath(__file__))
_HC0 = sum(E.Y / 12 * (1 + E.RF) ** -j for j in range(1, 12 * E.HC_YEARS + 1))
DELTA = E.EX_C - E.G / (1 + _HC0) * E.VC
CONTRACT_FRAC, ARITH_N, CONTRACT_CAP = 1 / 3, 36, 0.10
Q_RET, CS_ALPHA, CS_VOPT_N = 0.95, 0.05, 60
_CAL = None
import simkit_params as SP


def calib(which):
    global _CAL
    if _CAL is None:
        _CAL = SP.calib()
    d = _CAL[which]
    return np.array(d["mu"]), np.array(d["tau"]), np.array(d["omega"])


@dataclass(frozen=True)
class LWorld3:
    lw2: L2.LWorld2 = field(default_factory=L2.LWorld2)
    calib: str | None = None
    planted: tuple | None = None
    contract_frac: float = CONTRACT_FRAC


@dataclass(frozen=True)
class LArm3(L2.LArm2):
    group_sum: float | None = None
    contract: bool = False


ARMS = {
    "core": LArm3("core", core_only=True),
    "A": LArm3("A"),
    "Ep": LArm3("Ep", "family", group_sum=None),
    "Cp": LArm3("Cp", contract=True),
    "Rs": LArm3("Rs", retire="slab_delta"),
    "Rc": LArm3("Rc", retire="cs_delta"),
}


def draw3(lw3: LWorld3):
    lw = lw3.lw2.lw
    d = _draw_calib(lw3) if (lw3.calib or lw3.planted) else L2.draw2(lw3.lw2)
    if "fam" not in d:
        d["fam"] = np.arange(d["real"].shape[1]) // L2.FAM
    P, N = d["real"].shape
    rng5 = np.random.default_rng(lw.w.seed + 30_000)
    d["contract"] = rng5.random((P, N)) < lw3.contract_frac
    eta = rng5.normal(0, 1, (P, N))
    e_arr = np.take_along_axis(d["edge_live"], d["arrive"][:, :, None], 2)[:, :, 0]
    d["arith"] = e_arr + d["sd_obs"] / math.sqrt(ARITH_N) * eta
    return d


def _draw_calib(lw3: LWorld3):
\

    lw = lw3.lw2.lw
    w = lw.w
    assert lw.kind == "monthly" and lw.half_life is None and w.p_die == 0
    rng = np.random.default_rng(w.seed)
    P, N, T0, M = w.paths, w.n_rows, w.t0, E.MONTHS
    rng.random((P, N))
    shape = rng.choice(4, size=(P, N), p=np.array(w.mix))
    a = np.array([E.SHAPES[s][0] for s in E.SHAPE_NAMES])[shape]
    k = np.array([w.put_k if s == "put" else E.SHAPES[s][1] for s in E.SHAPE_NAMES])[shape]
    arrive = rng.integers(0, w.birth_window, (P, N))
    steps = T0 + M
    st = rng.random((P, N)) < w.p_revive / (w.p_die + w.p_revive)
    for s in range(steps):
        rng.random((P, N))
    assert st.all()
    core_pre = CC.draw_excess(rng, (P, T0))
    noise_pre = rng.normal(0, 1, (P, N, T0))
    core_live = CC.draw_excess(rng, (M, P))
    noise_live = rng.normal(0, 1, (M, P, N))
    rng2 = np.random.default_rng(w.seed + 10_000)
    has_replay = rng2.random((P, N)) < lw.replay_frac
    rng4 = np.random.default_rng(w.seed + 40_000)
    fn = N // L2.FAM
    if lw3.planted:
        mu = np.full(P, lw3.planted[0]); tau = np.full(P, lw3.planted[1]); om = np.full(P, 0.5)
    else:
        mus, taus, oms = calib(lw3.calib)
        i = rng4.integers(0, mus.size, P)
        mu, tau, om = mus[i], taus[i], oms[i]
    u = rng4.normal(0, 1, (P, fn)); e = rng4.normal(0, 1, (P, N))
    s_ann = mu[:, None] + (np.sqrt(om) * tau)[:, None] * np.repeat(u, L2.FAM, axis=1) + (np.sqrt(1 - om) * tau)[:, None] * e
    theta = s_ann / math.sqrt(12) * E.S
    edge = np.broadcast_to(theta[:, :, None], (P, N, steps))
    sd = E.S
    pre = edge[:, :, :T0] + E.shape_ret(core_pre[:, None, :], a[:, :, None], k[:, :, None]) + sd * noise_pre
    return dict(real=theta > 0, shape=shape, a=a, k=k, arrive=arrive, core_pre=core_pre, core_live=core_live,
                has_replay=has_replay, alive_live=np.ones((P, N, M), bool), pre=np.maximum(pre, -1.0),
                edge_live=np.ascontiguousarray(edge[:, :, T0:]), noise_live=sd * noise_live, sd_obs=sd, floor=-1.0,
                theta=theta)


def project_group(z, lo, hi, budget, g, gb, outer=50):
\

    x = E.project(z, lo, hi, budget)
    over = (x * g).sum(1) > gb
    if not over.any():
        return x
    zo, lo_o, hi_o, bo, go, gbo = z[over], lo[over], hi[over], budget[over], g[over].astype(float), gb[over]
    a = np.zeros(zo.shape[0]); b = np.maximum((zo * go - 1e300 * (1 - go)).max(1) - (lo_o * go).min(1), 0.0) + 1e-9
    for _ in range(outer):
        m = (a + b) / 2
        s = (E.project(zo - m[:, None] * go, lo_o, hi_o, bo) * go).sum(1)
        big = s > gbo
        a = np.where(big, m, a); b = np.where(big, b, m)
    x[over] = E.project(zo - b[:, None] * go, lo_o, hi_o, bo)
    return x


def solve_group(mu, bvec, dvec, lo, hi, budget, kap, x0, iters, g, gb):

    Lc = kap * (E.VC * (bvec ** 2).sum(1) + dvec.max(1)) + 1e-12
    step = (1 / Lc)[:, None]
    x = project_group(x0, lo, hi, budget, g, gb); yv = x.copy(); tk = 1.0
    for _ in range(iters):
        f = (bvec * yv).sum(1)
        gr = mu - kap[:, None] * (E.VC * bvec * f[:, None] + dvec * yv)
        xn = project_group(yv + step * gr, lo, hi, budget, g, gb)
        tn = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
        yv = xn + ((tk - 1) / tn) * (xn - x)
        x, tk = xn, tn
    return x


def _logndtr(u):
    if u > -30:
        return math.log(0.5 * math.erfc(-u / math.sqrt(2)))
    return -0.5 * u * u - math.log(-u) - 0.5 * math.log(2 * math.pi)


def cs_rho_ratio(alpha=CS_ALPHA):
    l = 2 * math.log(1 / (2 * alpha))
    return l + math.log(1 + l)


def cs_u(r, alpha=CS_ALPHA):

    c = math.log(1 / alpha) - math.log(2) + 0.5 * math.log1p(r)
    lo, hi = -50.0, 100.0
    for _ in range(200):
        m = (lo + hi) / 2
        if m * m / 2 + _logndtr(m) > c:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


def cs_boundary(V, v_opt, alpha=CS_ALPHA):
    rho = v_opt / cs_rho_ratio(alpha)
    return math.sqrt(V + rho) * cs_u(V / rho, alpha)


_U_TAB = None


def cs_ucb(n, rbar, se1):
\

    global _U_TAB
    if _U_TAB is None:
        ratio = cs_rho_ratio()
        _U_TAB = np.array([cs_u(nn * ratio / CS_VOPT_N) for nn in range(0, 4001)])
    nn = np.minimum(np.rint(n).astype(int), 4000)
    rho = CS_VOPT_N * se1 / cs_rho_ratio()
    B = np.sqrt(n * se1 + rho) * _U_TAB[nn]
    return rbar + B / np.maximum(n, 1)


def run3(lw3: LWorld3, arm: LArm3, d=None, iters=25):
    lw = lw3.lw2.lw
    d = d if d is not None else draw3(lw3)
    P, N = d["real"].shape
    assert lw.kind != "rare"
    fam_arm = arm.no_replay == "family"
    p, m1, v1, drop = L.priors(lw, d, arm)
    hr = d["has_replay"]
    so = d["sd_obs"]
    admitted = np.zeros((P, N), bool) if arm.core_only else (hr if arm.no_replay == "reject" else np.ones((P, N), bool))
    nr_ctr = np.zeros((P, N), bool)
    if arm.contract and not arm.core_only:
        nr_ctr = ~hr & d["contract"] & (d["arith"] > 0)
        q, es, vs = F.replay_prior(d["arith"] / so * math.sqrt(ARITH_N), ARITH_N)
        p = np.where(nr_ctr, q, p); m1 = np.where(nr_ctr, es * so, m1); v1 = np.where(nr_ctr, np.maximum(vs * so * so, 1e-10), v1)
        admitted = admitted | nr_ctr
    suspended = drop.copy()
    so2 = so * so
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
        ok = nb >= 6
        bhat = np.where(ok, (Sxy - Sx * Sy / np.maximum(nb, 1)) / np.maximum(Sxx - Sx * Sx / np.maximum(nb, 1), 1e-12), 0.0)
        se1 = so2 + bhat ** 2 * E.VC
        pt, m1t, v1t = p, m1, v1
        if fam_arm:
            q_o, m_o, v_o = F.slab_parts(nlive, ysum, se1, p, m1, v1)
            ev = (d["arrive"] <= t) & (hr | (nlive > 0))
            pf, mf, vf = L2.family_prior(q_o, m_o, v_o, ev, so)
            pt, m1t, v1t = np.where(hr, p, pf), np.where(hr, m1, mf), np.where(hr, v1, vf)
        mean, var, q = F.spike_slab(nlive, ysum, se1, pt, m1t, v1t)
        if arm.retire == "slab_delta":
            qs, ms, vs_ = F.slab_parts(nlive, ysum, se1, pt, m1t, v1t)
            p_le = (1 - qs) + qs * F.ndtr((DELTA - ms) / np.sqrt(vs_))
            dn = live & ~suspended & (nlive >= 1) & (p_le >= Q_RET)
            dead_by |= dn; suspended |= dn
        elif arm.retire == "cs_delta":
            ucb = cs_ucb(nlive, ysum / np.maximum(nlive, 1), se1)
            dn = live & ~suspended & (nlive >= 1) & (ucb <= DELTA)
            dead_by |= dn; suspended |= dn
        elif arm.retire == "floor" and arm.q_dead is not None:
            dn = live & ~suspended & (nlive >= arm.n_dead) & (q < arm.q_dead)
            dead_by |= dn; suspended |= dn
        sized = live & ~suspended & (nlive >= arm.shadow_k)
        risk = so2
        row_cap = np.where(nr_ctr, CONTRACT_CAP, arm.cap) if arm.contract else arm.cap
        mu = np.concatenate([np.full((P, 1), E.EX_C), mean], 1)
        bvec = np.concatenate([np.ones((P, 1)), bhat], 1)
        dvec = np.concatenate([np.zeros((P, 1)), risk + var], 1)
        hi = np.concatenate([W[:, None], np.where(sized, row_cap * W[:, None], 0.0)], 1)
        x0 = np.clip(x * (W / np.maximum(x.sum(1), 1e-12))[:, None] if t else np.c_[W, np.zeros((P, N))], 0, hi)
        it = iters if t else 4 * iters
        if arm.group_sum is not None:
            g = np.concatenate([np.zeros((P, 1), bool), sized & ~hr], 1)
            x = solve_group(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, x0, it, g, arm.group_sum * W)
        else:
            x = E.solve(mu, bvec, dvec, np.zeros_like(hi), hi, W.copy(), kap, x0, it, 0.0)
        hold = x[:, 1:]
        rc = d["core_live"][t]
        ry = np.maximum(d["edge_live"][:, :, t] + E.shape_ret(rc[:, None], d["a"], d["k"]) + d["noise_live"][t], d["floor"])
        cash = W - x[:, 0] - hold.sum(1)
        sat += (hold.sum(1) / W).mean() / E.MONTHS; fake += ((hold * ~d["real"]).sum(1) / W).mean() / E.MONTHS
        nrs += ((hold * ~hr).sum(1) / W).mean() / E.MONTHS
        W = np.maximum(x[:, 0] * (1 + E.RF + rc) + (hold * (1 + E.RF + ry)).sum(1) + cash * (1 + E.RF) + E.Y / 12, 1e-9)
        ob = live.astype(float)
        nlive += ob; ysum += ob * ry
        rcc = (rc - E.EX_C)[:, None]
        Sxy += ob * rcc * ry; Sxx = Sxx + ob * rcc ** 2; Sx = Sx + ob * rcc; Sy += ob * ry; nb = nb + ob
    real = d["real"]
    adm = admitted & ~drop
    info = dict(sat_share=sat, fake_share=fake, noreplay_share=nrs,
                retired_real=(dead_by & real).sum() / max((adm & real).sum(), 1),
                retired_fake=(dead_by & ~real).sum() / max((adm & ~real).sum(), 1),
                retired_all=dead_by.sum() / max(adm.sum(), 1),
                contract_admitted=nr_ctr.sum() / max((~hr & d["contract"]).sum(), 1),
                dropped_replay=drop.sum() / max(hr.sum(), 1))
    return W, info
