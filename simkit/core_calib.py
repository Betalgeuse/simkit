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

RF_A, PREM_A, SIGMA_A = 0.0245, 0.0299, 0.191
RF = RF_A / 12
PC, CR = 1 / 120, -0.15
EX_TARGET = PREM_A / 12
VAR_TARGET = SIGMA_A ** 2 / 12
VC_JUMP = PC * CR * CR - (PC * CR) ** 2
_M_LN = EX_TARGET + RF - PC * CR
SD_C = math.sqrt(math.log(1 + (VAR_TARGET - VC_JUMP) / (1 + _M_LN) ** 2))
MU_C = math.log(1 + _M_LN) - SD_C ** 2 / 2


def draw_excess(rng, shape):

    return np.exp(rng.normal(MU_C, SD_C, shape)) - 1 + np.where(rng.random(shape) < PC, CR, 0.0) - RF


def assert_matches_stated(draw, n=4_000_000, batches=200, z=4.5, seed=12345):
\

    rng = np.random.default_rng(seed)
    x = draw(rng, n).reshape(batches, -1)
    bm, bs = x.mean(1), x.std(1)
    m, s = bm.mean(), bs.mean()
    se_m, se_s = bm.std(ddof=1) / math.sqrt(batches), bs.std(ddof=1) / math.sqrt(batches)
    assert abs(m - EX_TARGET) < z * se_m, f"core mean {m*12:.4%}/yr vs stated {PREM_A:.2%} (se {se_m*12:.4%})"
    assert abs(s - SIGMA_A / math.sqrt(12)) < z * se_s, f"core vol {s*math.sqrt(12):.4f} vs stated {SIGMA_A} (se {se_s*math.sqrt(12):.4f})"
    return m * 12, s * math.sqrt(12), se_m * 12, se_s * math.sqrt(12)
