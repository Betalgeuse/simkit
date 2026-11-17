# simkit

Monte Carlo engines for sizing many uncertain return sources inside one CRRA portfolio: a lognormal core with crashes,
candidate rows that are real or fake and switch alive/dead, admission rules, empirical-Bayes pooling, a hidden-Markov
edge filter and a budget-constrained CRRA allocator. Every world here is synthetic.

    pip install -r requirements.txt
    pytest -q
    cd simkit && python run_cell.py base 1 2000 CUR,NEW025,NEW11 ../out

`simkit_params.py` holds every run parameter with a synthetic default. `SIMKIT_PARAMS=<file.json>` overrides them for a
local run. The `sims` workflow runs a world x seed matrix on the defaults only.
