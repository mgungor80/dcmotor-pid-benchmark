"""5 optimisers x 30 runs on the journal formulation (x = Kp, Ki, Kd, N; Ms <= 1.4; Mu <= 150; 12 V; 1 ms)."""
import sys, json, time, numpy as np
from multiprocessing import Pool
from formulation import Problem, BOUNDS
from algorithms import de, pso, ga, gwo, hho
ALGS = {'DE': de, 'PSO': pso, 'GA': ga, 'GWO': gwo, 'HHO': hho}
LB = np.array([b[0] for b in BOUNDS], float); UB = np.array([b[1] for b in BOUNDS], float)
NPOP = 30; RUNS = 30; BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
_P = None
def job(a):
    global _P
    if _P is None: _P = Problem()
    n, r = a; t0 = time.time()
    x, f, c = ALGS[n](_P.cost, LB, UB, NPOP, BUDGET, np.random.default_rng(1000 + r))
    return n, r, np.asarray(x).tolist(), float(f), np.asarray(c).tolist(), time.time() - t0
if __name__ == '__main__':
    with Pool(2) as p: res = p.map(job, [(n, r) for n in ALGS for r in range(RUNS)], chunksize=3)
    out = {n: dict(F=[0]*RUNS, X=[0]*RUNS, curve=[0]*RUNS, time=[0]*RUNS) for n in ALGS}
    for n, r, x, f, c, tm in res: out[n]['F'][r] = f; out[n]['X'][r] = x; out[n]['curve'][r] = c; out[n]['time'][r] = tm
    json.dump(dict(budget=BUDGET, N=NPOP, runs=RUNS, lb=LB.tolist(), ub=UB.tolist(), res=out), open(f'alg2_{BUDGET}.json', 'w'))
    for n in ALGS:
        F = np.array(out[n]['F']); print(f'{n:4s} best {F.min():.6f} mean {F.mean():.6f} med {np.median(F):.6f} worst {F.max():.6f} t {np.mean(out[n]["time"]):.1f}s')
