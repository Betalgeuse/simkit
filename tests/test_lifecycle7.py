

import numpy as np
import b2_engine as E
import lifecycle7_engine as A7

def test_eb_recovers_population():
    rng = np.random.default_rng(1)
    P, N = 50, 400
    real = rng.random((P, N)) < 0.3
    theta = np.where(real, 0.004 + 0.002 * rng.standard_normal((P, N)), 0.0)
    se2 = np.full((P, N), 0.002 ** 2)
    z = theta + np.sqrt(se2) * rng.standard_normal((P, N))
    pi1, m, tau2 = A7.eb_fit(z, se2, np.ones((P, N), bool), iters=200)
    assert abs(pi1.mean() - 0.3) < 0.05, pi1.mean()
    assert abs(m.mean() - 0.004) < 0.0007, m.mean()
    assert abs(np.sqrt(tau2).mean() - 0.002) < 0.0007, np.sqrt(tau2).mean()

def test_posterior_calibrated():

    rng = np.random.default_rng(2)
    P, N = 40, 400
    real = rng.random((P, N)) < 0.2
    theta = np.where(real, 0.003 + 0.002 * rng.standard_normal((P, N)), 0.0)
    n = np.full((P, N), 120.0); se1 = np.full((P, N), 0.03 ** 2)
    s = n * (theta + np.sqrt(se1 / n) * rng.standard_normal((P, N)))
    q, mp, vp, _ = A7.eb_posterior(n, s, se1, np.ones((P, N), bool), 0.0, 1e-6)
    pm = (q * mp).ravel(); th = theta.ravel()
    slope = np.cov(pm, th)[0, 1] / pm.var()
    assert 0.85 < slope < 1.15, slope

def _one(mu, sd, fat=False, bd=0.0, q=1.0, vp=0.0):
    P, N = 4, 1
    rng = np.random.default_rng(3)
    A7.S_SCEN = 4000
    c, r = A7.scenarios(rng, np.full((P, N), q), np.full((P, N), mu), np.full((P, N), vp), np.zeros((P, N)),
                        np.full((P, N), bd), np.full((P, N), sd), np.ones((P, N), bool), fat, np.ones((P, N)), P, N)
    c = c * 0
    W = np.ones(P); HC = np.zeros(P)
    x = A7.ce_alloc(c, r, np.ones((P, N), bool), W, HC, np.c_[W, np.zeros((P, N))], 400, np.full(P, 2.0 * sd * sd * 2))
    A7.S_SCEN = 64
    return x[:, 1].mean()

def test_merton_closed_form():
    w = _one(0.002, 0.05)
    assert abs(w - 0.002 / (2 * 0.05 ** 2)) < 0.12, w

def test_zero_edge_zero_weight():
    assert _one(0.0, 0.05) < 0.02

def test_parameter_uncertainty_lowers_weight():
    assert _one(0.002, 0.05, vp=0.03 ** 2) < _one(0.002, 0.05) - 0.05

def test_fdr_cuts_weight():
    assert _one(0.002, 0.05, q=0.3) < 0.5 * _one(0.002, 0.05)

def test_downside_beta_lowers_weight():
    assert _one(0.002, 0.05, bd=0.5) < _one(0.002, 0.05) - 0.02

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
