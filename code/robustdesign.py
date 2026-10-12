"""Scenario-based robust design (set B, all six parameters +-20 %): 40 training plants (+ nominal).
min mean J over training plants  s.t.  max Ms <= 1.4, max Mu <= Mu_max, all plants stable."""
import json, sys, numpy as np
from motor import MOTOR
from sim4 import Sim4
from montecarlo import Gz, Cz, cl_stable
KEYS = ('Ra','La','J','B','Kt','Kb'); NT = 40
F = np.vstack([np.ones(6), np.random.default_rng(12345).uniform(0.8, 1.2, (NT, 6))])   # training seed != test seed (7)
SIMS = []
for f in F:
    m = dict(MOTOR); m.update({k: MOTOR[k]*fi for k, fi in zip(KEYS, f)}); s = Sim4(motor=m); SIMS.append(s)
GS = np.array([Gz(s) for s in SIMS])
def margins(K, N):
    C = Cz(K, N); S = 1/(1+C*GS); return np.abs(S).max(), np.abs(C*S).max()
def cost(x, Mu_max, w=10.0):
    K, N = x[:3], x[3]
    Ms, Mu = margins(K, N)
    pen = w*max(0, Ms-1.4) + w*max(0, Mu/Mu_max-1)
    if pen > 0.05: return 1.0 + pen                      # far infeasible: skip simulation
    if not all(cl_stable(s, K, N) for s in SIMS): return 1e6
    J = np.mean([s.run(K, dom=1, N=N, out=False) + s.run(K, r=0, TL=1e-3, dom=1, N=N, out=False) for s in SIMS])
    return J + pen
if __name__ == '__main__':
    from scipy.optimize import differential_evolution
    out = {}
    for Mu_max in [float(a) for a in sys.argv[1:]]:
        best = None
        for seed in (1, 2):
            r = differential_evolution(cost, [(0,100),(0,100),(0,10),(5,500)], args=(Mu_max,), seed=seed, popsize=20,
                                       maxiter=400, tol=1e-12, polish=False, mutation=(0.5,1), recombination=0.9)
            if best is None or r.fun < best.fun: best = r
        K, N = best.x[:3], best.x[3]; Ms, Mu = margins(K, N)
        out[Mu_max] = dict(x=best.x.tolist(), J=best.fun, Ms=float(Ms), Mu=float(Mu))
        print(f'ROBUST Mu_max={Mu_max:6.1f} K=({K[0]:.3f},{K[1]:.4f},{K[2]:.4f}) N={N:.1f} meanJ={best.fun:.6f} MsWC={Ms:.3f} MuWC={Mu:.1f}', flush=True)
    json.dump(out, open('robust_%s.json' % '_'.join(sys.argv[1:]), 'w'))
