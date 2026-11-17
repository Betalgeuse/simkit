
\
\
\
\
\
\
\
\
\

import numpy as np
import b2_engine as E
import lifecycle3_engine as L3

_project_group_bisect = L3.project_group


def project_group_exact(z, lo, hi, budget, g, gb, outer=None):
    x = E.project(z, lo, hi, budget)
    over = (x * g).sum(1) > gb
    if not over.any():
        return x
    assert not lo.any(), "exact group projection assumes lo = 0"
    zo, lo_o, hi_o, bo, go, gbo = z[over], lo[over], hi[over], budget[over], g[over], gb[over]
    xg = E.project(zo, lo_o, np.where(go, hi_o, 0.0), gbo)
    xr = E.project(zo, lo_o, np.where(go, 0.0, hi_o), bo - xg.sum(1))
    x[over] = xg + xr
    return x


L3.project_group = project_group_exact
