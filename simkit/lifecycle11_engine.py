
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
import numpy as np
import b2_engine as E
import core_calib as CC
import lifecycle_engine as L
import lifecycle2_engine as L2
import lifecycle7_engine as A7
import lifecycle5_engine as A5

S_SCEN = A7.S_SCEN
GAMMA = E.G
EM_ITERS = 3
REFIT_EVERY = 12
N0_BD = 60.0
N0_S2 = 24.0
import os
KNOWN_SD = os.environ.get('L8_KNOWN_SD') == '1'
STATE_WIN = 36
_r = CC.draw_excess(np.random.default_rng(1), 2_000_000) - E.EX_C
EM_C = float(np.minimum(_r, 0.0).mean())


def kink_mean(kink):

    return float(np.minimum(_r + E.EX_C - kink, 0.0).mean())


def strat_idx(rng, M, P, S, N):
\
\
\

    k, rem = divmod(S, M)
    parts = [np.broadcast_to(np.arange(M), (P, N, M))] * k
    if rem:
        parts.append(np.argsort(rng.random((P, N, M)), -1)[..., :rem])
    idx = np.concatenate(parts, -1)
    perm = np.argsort(rng.random((P, N, S)), -1)
    return np.take_along_axis(idx, perm, -1).transpose(0, 2, 1)


def trans(pd, pr):

    return np.array([[1, 0, 0], [0, 1 - pr, pr], [0, pd, 1 - pd]], float)


def init_vec(pi, pd, pr):
    sa = pr / (pd + pr)
    pi = np.asarray(pi)[..., None]
    return np.concatenate([1 - pi, pi * (1 - sa), pi * sa], -1)


def emis(u, s2, mu_a, obs):

    l0 = np.exp(-0.5 * u * u / s2)
    la = np.exp(-0.5 * (u - mu_a) ** 2 / s2)
    lik = np.stack([l0, l0, la], -1)
    return np.where(obs[..., None], lik, 1.0)


def forward(U, S2, MU, OBS, a0, T3):
\

    P, N, T = U.shape
    pred = np.empty((P, N, T + 1, 3)); filt = np.empty((P, N, T, 3))
    a = a0
    for t in range(T):
        pred[:, :, t] = a
        f = a * emis(U[:, :, t], S2[:, :, t], MU[:, :, t], OBS[:, :, t])
        f /= np.maximum(f.sum(-1, keepdims=True), 1e-300)
        filt[:, :, t] = f
        a = f @ T3
    pred[:, :, T] = a
    return pred, filt


def em_fit(U, S2, OBS, UP, pi, e, pd, pr, state, iters=EM_ITERS):
\

    P, N, T = U.shape
    T3 = trans(pd, pr)
    anyobs = OBS.any(2)
    for _ in range(iters):
        MU = np.where(UP[:, None, :], e[:, 0, None, None], e[:, 1, None, None])
        pred, filt = forward(U, S2, MU, OBS, init_vec(np.broadcast_to(pi[:, None], (P, N)), pd, pr), T3)
        b = np.ones((P, N, 3))
        su = np.zeros((P, 2)); sw = np.zeros((P, 2)); gfake = np.zeros((P, N))
        for t in range(T - 1, -1, -1):
            g = filt[:, :, t] * b
            g /= np.maximum(g.sum(-1, keepdims=True), 1e-300)
            ga = np.where(OBS[:, :, t], g[..., 2], 0.0) / S2[:, :, t]
            k = UP[:, t].astype(int)
            col = 1 - k
            wsum = ga.sum(1); usum = (ga * U[:, :, t]).sum(1)
            sw[np.arange(P), col] += wsum; su[np.arange(P), col] += usum
            if t == 0:
                gfake = g[..., 0]
            lik = emis(U[:, :, t], S2[:, :, t], MU[:, :, t], OBS[:, :, t])
            b = (lik * b) @ T3.T
            b /= np.maximum(b.sum(-1, keepdims=True), 1e-300)
        pi = np.clip((((1 - gfake) * anyobs).sum(1) + 1) / (anyobs.sum(1) + 2), 1e-3, 1 - 1e-3)
        if state:
            e = np.where(sw > 1e-9, su / np.maximum(sw, 1e-300), e)
        else:
            tot = np.where(sw.sum(1) > 1e-9, su.sum(1) / np.maximum(sw.sum(1), 1e-300), e[:, 0])
            e = np.stack([tot, tot], 1); sw = np.stack([sw.sum(1)] * 2, 1)
    var_e = 1 / np.maximum(sw, 1e-300)
    return pi, e, np.minimum(var_e, 1.0)


def scenarios8(rng, q, e, ve, b, bd, sd, sized, fat, rare_scale, P, N):
\

    h = S_SCEN // 2
    anti = lambda z: np.concatenate([z, -z], 1)
    c = math.sqrt(E.VC) * anti(rng.standard_normal((P, h)))
    alive = np.tile(rng.random((P, h, N)) < q[:, None, :], (1, 2, 1))
    ed = e[:, None, None] + np.sqrt(ve)[:, None, None] * anti(rng.standard_normal((P, h, 1)))
    th = np.where(alive, ed, 0.0)
    eps = A7.EVENT_STD[rng.integers(0, A7.EVENT_STD.size, (P, S_SCEN, N))] if fat else anti(rng.standard_normal((P, h, N)))
    r = th + b[:, None, :] * c[:, :, None] + bd[:, None, :] * (np.minimum(c, 0.0) - EM_C)[:, :, None] \
        + (sd * rare_scale)[:, None, :] * eps
    return c + E.EX_C, np.where(sized[:, None, :], r, 0.0)


def market_up(core_hist):

    P, T = core_hist.shape
    cs = np.concatenate([np.zeros((P, 1)), np.cumsum(core_hist, 1)], 1)
    lo = np.maximum(np.arange(T) - STATE_WIN, 0)
    return (cs[:, np.arange(T)] - cs[:, lo]) >= 0


def state_world(d, seed, paths, n_rows, t0, w, mult_dn=-0.4):
\

    rng = np.random.default_rng(seed)
    P, N, T0, M = paths, n_rows, t0, E.MONTHS
    real = rng.random((P, N)) < w.p_real
    rng.choice(4, size=(P, N), p=np.array(w.mix)); rng.integers(0, w.birth_window, (P, N))
    st = rng.random((P, N)) < w.p_revive / (w.p_die + w.p_revive)
    alive = np.zeros((P, N, T0 + M), bool)
    for s in range(T0 + M):
        alive[:, :, s] = st
        u = rng.random((P, N)); st = np.where(st, u >= w.p_die, u < w.p_revive)
    edge = np.where(real[:, :, None] & alive, w.edge, 0.0)
    assert np.allclose(edge[:, :, T0:], d["edge_live"])
    up = market_up(np.concatenate([d["core_pre"], d["core_live"].T], 1))
    mult = np.where(up, 1.0, mult_dn)[:, None, :]
    d = dict(d)
    d["pre"] = np.maximum(d["pre"] + (mult[:, :, :T0] - 1) * edge[:, :, :T0], -1.0)
    d["edge_live"] = edge[:, :, T0:] * mult[:, :, T0:]
    return d


E_GRID = np.linspace(0.0, 0.01, 9)
PI_GRID = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
E_DN_GRID = np.array([-0.004, -0.002, 0.0, 0.002, 0.005])


def make_grid(state, scale=1.0, e_grid=None):
\

    if state:
        g = [(p, eu, ed) for p in PI_GRID[[0, 2, 4]] for eu in np.linspace(0, 0.01, 7) for ed in E_DN_GRID]
    else:
        g = [(p, e, e) for p in PI_GRID for e in (E_GRID if e_grid is None else e_grid)]
    a = np.array(g, float)
    a[:, 1:] *= scale
    return a


def KZ(mean, var, sized):
    num = np.where(sized, mean * mean, 0.0).sum(1)
    den = np.where(sized, mean * mean + np.maximum(var, 0.0), 0.0).sum(1)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def run8(lw2: L2.LWorld2, d=None, iters=25, crisis=False, state=False, state_world_on=False, hazard=None, cal_out=None,
         grid_scale=None, grid=None, e_grid=None, s_scen=None, beta_min_n=6, beta_zero=False,
         kink=None, beta_var=False, b_shrink=False, scen="iid", alloc="ce", pit_out=None):
\
\
\
\
\
\
\

    if kink is None:
        zf = lambda cc: np.minimum(cc, 0.0) - EM_C
    else:
        _km = kink_mean(kink)
        zf = lambda cc: np.minimum(cc + E.EX_C - kink, 0.0) - _km
    lw = lw2.lw
    w = lw.w
    d = d if d is not None else L2.draw2(lw2)
    if state_world_on:
        d = state_world(d, w.seed, w.paths, w.n_rows, w.t0, w)
    P, N = d["real"].shape
    rare = lw.kind == "rare"
    fat = lw.kind in ("fat", "rare")
    pd, pr = hazard if hazard else (w.p_die, w.p_revive)
    T3 = trans(pd, pr)
    hr = d["has_replay"]
    so2 = d["sd_obs"] * d["sd_obs"]
    srng = np.random.default_rng(w.seed + 80_000)
    M = E.MONTHS
    if crisis:
        cpd = np.minimum(d["core_pre"] - E.EX_C, 0.0)[:, None, :]
        d = dict(d); d["pre"] = np.maximum(d["pre"] + A7.CRISIS_BD * cpd, -1.0)

    gs = grid_scale if grid_scale else (1.0 if not fat else d["sd_obs"] / E.S if not rare else (L.EVENT_EDGE / L.HOLD) / 0.005)
    GR = make_grid(state, gs, e_grid) if grid is None else np.asarray(grid, float)
    S = s_scen or S_SCEN
    G = GR.shape[0]
    gpi, geu, ged = GR[:, 0], GR[:, 1], GR[:, 2]
    ll = np.zeros((G, P))
    full = np.concatenate([d["core_pre"], d["core_live"].T], 1)
    T0p = d["core_pre"].shape[1]
    UPfull = market_up(full)
    W = np.ones(P); x = np.zeros((P, N + 1)); ks = []
    sat = fake = nsz = 0.0
    cal = np.zeros(5); hyp = np.zeros(3)
    b = np.zeros((P, N)); bd = np.zeros((P, N)); s2r = np.full((P, N), so2)
    Vb = np.zeros((P, N, 2, 2))

    Gm = np.zeros((P, N, 3, 3)); hv = np.zeros((P, N, 3)); Syy = np.zeros((P, N))

    def acc(y, c, ob):
        nonlocal Syy
        xs = [np.ones_like(c), c, zf(c)]
        for i in range(3):
            hv[..., i] += ob * xs[i] * y
            for j in range(3):
                Gm[..., i, j] += ob * xs[i] * xs[j]
        Syy += ob * y * y

    def fit_betas():
        nonlocal b, bd, s2r, Vb
        n = Gm[..., 0, 0]
        Ginv = np.linalg.inv(Gm + 1e-9 * np.eye(3))
        beta = (Ginv @ hv[..., None])[..., 0]
        ok = n >= beta_min_n
        rss = np.maximum(Syy - (beta * hv).sum(-1), 0.0)


        dfree = np.maximum(n - 3, 0)
        pool = (rss.sum(1) + N0_S2 * so2) / (dfree.sum(1) + N0_S2)
        s2r = (rss + N0_S2 * pool[:, None]) / (dfree + N0_S2)
        if KNOWN_SD:
            s2r = np.full((P, N), so2)
        okf = ok.astype(float)
        bd_pop = (np.where(ok, beta[..., 2], 0.0) * okf).sum(1) / np.maximum(okf.sum(1), 1)
        sh = n / (n + N0_BD)
        bd = np.where(ok, sh * beta[..., 2] + (1 - sh) * bd_pop[:, None], bd_pop[:, None])
        b = np.where(ok, beta[..., 1], 0.0)
        if b_shrink:
            b_pop = (np.where(ok, beta[..., 1], 0.0) * okf).sum(1) / np.maximum(okf.sum(1), 1)
            b = np.where(ok, sh * beta[..., 1] + (1 - sh) * b_pop[:, None], b_pop[:, None])
        if beta_var:
            wv = np.stack([sh if b_shrink else np.ones_like(sh), sh], -1)
            V = s2r[..., None, None] * Ginv[..., 1:, 1:] * wv[..., :, None] * wv[..., None, :]
            bt = np.stack([beta[..., 1], beta[..., 2]], -1)
            mu_ok = (bt * okf[..., None]).sum(1) / np.maximum(okf.sum(1), 1)[:, None]
            vb_between = (((bt - mu_ok[:, None]) ** 2) * okf[..., None]).sum(1) / np.maximum(okf.sum(1) - 1, 1)[:, None]
            Vfall = np.zeros_like(V); Vfall[..., 0, 0] = vb_between[:, None, 0]; Vfall[..., 1, 1] = vb_between[:, None, 1] * (1 - sh) ** 2
            Vb = np.where(ok[..., None, None], V, Vfall)
        if beta_zero:
            b, bd = np.zeros((P, N)), np.zeros((P, N))

    def s2_at(cc):

        if not beta_var:
            return s2r
        z = zf(cc)
        return s2r + cc * cc * Vb[..., 0, 0] + 2 * cc * z * Vb[..., 0, 1] + z * z * Vb[..., 1, 1]

    def mu_g(tidx):

        up = UPfull[:, tidx] if state else np.ones(P, bool)
        return np.where(up[None, :], geu[:, None], ged[:, None])

    def step(a, u, s2, mu, ob):

        l0 = np.exp(-0.5 * u * u / s2)[None]
        la = np.exp(-0.5 * (u[None] - mu[:, :, None]) ** 2 / s2[None])
        lik = np.stack([np.broadcast_to(l0, la.shape), np.broadcast_to(l0, la.shape), la], -1)
        lik = np.where(ob[None, :, :, None], lik, 1.0)
        f = a * lik
        z = f.sum(-1)
        ll[:] += np.where(ob[None], np.log(np.maximum(z, 1e-300)), 0.0).sum(2)
        f /= np.maximum(z[..., None], 1e-300)
        return f @ T3

    sa = pr / (pd + pr)
    a = init_vec(np.broadcast_to(gpi[:, None, None], (G, P, N)), pd, pr)
    if rare:
        sc2 = so2
        n0 = np.where(hr, d["pre_n"], 0.0); s0_ = np.where(hr, d["pre_s1"], 0.0)
        v0 = np.maximum(n0 * sc2, 1e-300)
        l1 = np.exp(-0.5 * (s0_[None] - n0[None] * sa * geu[:, None, None]) ** 2 / v0[None])
        l0 = np.exp(-0.5 * s0_ ** 2 / v0)[None]
        ll += np.log(np.maximum(gpi[:, None, None] * l1 + (1 - gpi[:, None, None]) * l0, 1e-300)).sum(2)
        preal = gpi[:, None, None] * l1 / np.maximum(gpi[:, None, None] * l1 + (1 - gpi[:, None, None]) * l0, 1e-300)
        a = init_vec(preal, pd, pr)
        NC_lag = None
    else:
        pre = d["pre"]; cpre = d["core_pre"] - E.EX_C
        for s in range(T0p):
            acc(pre[:, :, s], cpre[:, s, None], hr.astype(float))
        fit_betas()
        for s in range(T0p):
            u = pre[:, :, s] - b * cpre[:, s, None] - bd * zf(cpre[:, s, None])
            a = step(a, u, s2_at(cpre[:, s, None]), mu_g(s), hr)
    for t in range(M):
        tn = T0p + t
        rem = max(0, 12 * E.HC_YEARS - t)
        HC = sum(E.Y / 12 * (1 + E.RF) ** -j for j in range(1, rem + 1)) if rem else 0.0
        live = d["arrive"] <= t
        wg = np.exp(ll - ll.max(0)); wg /= wg.sum(0)
        if rare:
            lo_c = np.maximum(d["arrive"], t - L.HOLD + 1)
            mm = np.arange(M)[None, None, :]
            open_mask = (mm >= lo_c[:, :, None]) & (mm <= t)
            n_open = (d["c"] * open_mask).sum(2); s_open = (d["s1"] * open_mask).sum(2)
            qg = (a @ np.linalg.matrix_power(T3, L.HOLD // 2))[..., 2]
            sized = live & (n_open > 0)
            sd = np.full((P, N), math.sqrt(so2)); scale = np.sqrt(L.HOLD / np.maximum(n_open, 1))
            bb = np.zeros((P, N)); bdd = np.zeros((P, N))
        else:
            qg = a[..., 2]
            sized = live
            sd = np.sqrt(s2r); scale = np.ones((P, N)); bb, bdd = b, bd
        mg = mu_g(min(tn, UPfull.shape[1] - 1))
        mean8 = np.einsum("gp,gpn->pn", wg * mg, qg)
        qa = np.einsum("gp,gpn->pn", wg, qg)
        hyp += np.array([(wg * gpi[:, None]).sum(0).mean(), (wg * mg).sum(0).mean(), qa[live].mean() if live.any() else 0.0]) / M
        it = iters if t else 4 * iters
        x0 = np.clip(x * (W / np.maximum(x.sum(1), 1e-12))[:, None], 0, None) if t else np.c_[W, np.zeros((P, N))]

        h = S // 2
        cdf = np.cumsum(wg, 0)
        gi = np.minimum((srng.random((P, h))[None] > cdf[:, :, None]).sum(0), G - 1)
        pidx = np.arange(P)[:, None]
        qs = qg[gi, pidx]
        ms = mg[gi, pidx]
        anti = lambda z: np.concatenate([z, -z], 1)
        c = math.sqrt(E.VC) * anti(srng.standard_normal((P, h)))
        alive = np.tile(srng.random((P, h, N)) < qs, (1, 2, 1))
        th = np.where(alive, np.tile(ms, (1, 2))[:, :, None], 0.0)
        if fat and scen == "strat":
            eps = A7.EVENT_STD[strat_idx(srng, A7.EVENT_STD.size, P, S, N)]
        else:
            eps = A7.EVENT_STD[srng.integers(0, A7.EVENT_STD.size, (P, S, N))] if fat else anti(srng.standard_normal((P, h, N)))
            if fat and scen == "center":
                eps = eps - eps.mean(1, keepdims=True)
        bS, bdS = bb[:, None, :], bdd[:, None, :]
        if beta_var and not rare:
            bS = bS + np.sqrt(Vb[..., 0, 0])[:, None, :] * anti(srng.standard_normal((P, h, N)))
            bdS = bdS + np.sqrt(Vb[..., 1, 1])[:, None, :] * anti(srng.standard_normal((P, h, N)))
        r_s = th + bS * c[:, :, None] + bdS * zf(c)[:, :, None] + (sd * scale)[:, None, :] * eps
        r_s = np.where(sized[:, None, :], r_s, 0.0)
        c_s = c + E.EX_C
        dvec = sd ** 2 * scale ** 2 + np.abs(mg).max(0)[:, None] ** 2 + (bb ** 2 + bdd ** 2) * E.VC
        Lc = GAMMA / (W + HC) * (E.VC * 4 + np.where(sized, dvec, 0.0).max(1) * 2) + 1e-12
        if alloc == "new":
            var8 = np.maximum(np.einsum("gp,gpn->pn", wg * mg * mg, qg) - mean8 ** 2, 1e-12)
            risk = sd ** 2 * scale ** 2 + (Vb[..., 0, 0] * E.VC if beta_var and not rare else 0.0)
            kz = KZ(mean8, var8, sized)
            ks.append(float(kz[sized.any(1)].mean()) if sized.any() else float("nan"))
            x = A5.allocate(mean8, var8, risk, bb, d["fam"], sized, W, GAMMA / (W + HC), kz[:, None], x0, it)
        else:
            x = A7.ce_alloc(c_s, r_s, sized, W, HC, x0, it, Lc)
        if not rare:
            tr = d["edge_live"][:, :, t]
            xs_, ys_ = mean8[live], tr[live]
            cal += [xs_.size, xs_.sum(), ys_.sum(), (xs_ * ys_).sum(), (xs_ * xs_).sum()]
            if pit_out is not None:
                mq = wg[:, :, None] * qg; val = mg[:, :, None]; p0 = 1.0 - qa; tol = 1e-9 * max(float(np.abs(mg).max()), 1e-12)
                below = (mq * (val < tr[None] - tol)).sum(0) + p0 * (tr > tol)
                at = (mq * (np.abs(val - tr[None]) <= tol)).sum(0) + p0 * (np.abs(tr) <= tol)
                pit_out.append((below, at, mean8.copy(), tr.copy(), live.copy()))
            if cal_out is not None:
                cal_out.append((mean8[live].astype(np.float32), tr[live].astype(np.float32), qa[live].astype(np.float32),
                                d["alive_live"][:, :, t][live]))
        hold = x[:, 1:]
        rc = d["core_live"][t]
        if rare:
            ry = np.where(n_open > 0, s_open / np.maximum(n_open, 1), 0.0)
        else:
            ry = np.maximum(d["edge_live"][:, :, t] + E.shape_ret(rc[:, None], d["a"], d["k"]) + d["noise_live"][t], d["floor"])
            if crisis:
                ry = np.maximum(ry + A7.CRISIS_BD * np.minimum(rc - E.EX_C, 0.0)[:, None], d["floor"])
        cash = W - x[:, 0] - hold.sum(1)
        sat += (hold.sum(1) / W).mean() / M; fake += ((hold * ~d["real"]).sum(1) / W).mean() / M
        nsz += (hold > 1e-6 * W[:, None]).sum(1).mean() / M
        W = np.maximum(x[:, 0] * (1 + E.RF + rc) + (hold * (1 + E.RF + ry)).sum(1) + cash * (1 + E.RF) + E.Y / 12, 1e-9)
        if rare:
            s0 = t - L.HOLD + 1
            if s0 >= 0:
                obs = live & (d["arrive"] <= s0) & (d["c"][:, :, s0] > 0)
                ncs = np.maximum(d["c"][:, :, s0], 1)
                un = np.where(obs, d["s1"][:, :, s0] / ncs, 0.0)
                a = step(a, un, sc2 / ncs, mu_g(T0p + s0), obs)
            else:
                a = a @ T3
        else:
            rcc = rc - E.EX_C
            u = ry - b * rcc[:, None] - bd * zf(rcc[:, None])
            a = step(a, u, s2_at(rcc[:, None]), mu_g(tn), live)
            acc(ry, rcc[:, None], live.astype(float))
            if (t + 1) % REFIT_EVERY == 0:
                fit_betas()
    info = dict(sat_share=sat, fake_share=fake, rows_held=nsz, eb_pi=hyp[0], eb_e=hyp[1], q_alive=hyp[2],
                k_mean=float(np.nanmean(ks)) if ks else float("nan"))
    if cal_out is not None:
        wg = np.exp(ll - ll.max(0)); wg /= wg.sum(0)
        cal_out.append(("hyper", wg, GR))
    n_, sx, sy, sxy, sxx = cal
    if n_ > 100:
        info["cal8_slope"] = (sxy - sx * sy / n_) / max(sxx - sx * sx / n_, 1e-300)
    return W, info
