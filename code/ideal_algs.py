"""Five optimizers x 30 runs on the idealized benchmark (no saturation, D on error, N = 100, ITAE over 1 s,
box [0.001, 20]^3), same settings and seeds as run_algs2.py. Used to compare optimizer rankings between formulations."""
import sys, json, numpy as np
from multiprocessing import Pool
from scipy.stats import friedmanchisquare, rankdata
from motor import Motor
from algorithms import de, pso, ga, gwo, hho
ALGS = {'DE': de, 'PSO': pso, 'GA': ga, 'GWO': gwo, 'HHO': hho}
LB, UB = np.full(3, 0.001), np.full(3, 20.0); NPOP, RUNS = 30, 30
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
JREF = json.load(open('ideal_bound.json'))['20']['J']
_M = None
def f(x):
    t, y, u, J = _M.response(np.asarray(x, float)); return J if np.isfinite(J) else 1e6
def job(a):
    global _M
    if _M is None: _M = Motor(umax=0, Ts=1e-4, T=1.0)
    n, r = a; x, F, c = ALGS[n](f, LB, UB, NPOP, BUDGET, np.random.default_rng(1000 + r))
    return n, r, np.asarray(x).tolist(), float(F)
if __name__ == '__main__':
    with Pool(2) as p: res = p.map(job, [(n, r) for n in ALGS for r in range(RUNS)], chunksize=5)
    out = {n: dict(F=[0]*RUNS, X=[0]*RUNS) for n in ALGS}
    for n, r, x, F in res: out[n]['F'][r] = F; out[n]['X'][r] = x
    A = np.array([out[n]['F'] for n in ALGS]).T
    ranks = np.mean([rankdata(row) for row in A], 0); chi = friedmanchisquare(*A.T)
    summ = {n: dict(best=min(out[n]['F']) / JREF, median=float(np.median(out[n]['F'])) / JREF,
                    success=int(sum(F <= 1.01 * JREF for F in out[n]['F'])), rank=float(ranks[i]),
                    at_bound=int(sum(abs(x[0] - 20) < 1e-3 for x in out[n]['X']))) for i, n in enumerate(ALGS)}
    json.dump(dict(budget=BUDGET, Jref=JREF, friedman=[chi.statistic, chi.pvalue], summary=summ, res=out), open(f'ideal_algs_{BUDGET}.json', 'w'), indent=1)
    print(BUDGET, 'Friedman', chi.statistic, chi.pvalue)
    for n, s in summ.items(): print(n, {k: round(v, 4) for k, v in s.items()})
