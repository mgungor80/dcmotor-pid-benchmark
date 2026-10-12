"""Idealised benchmark (no saturation, Ts = 0.1 ms, D on error, N = 100, ITAE over 1 s): optimum vs box size."""
import json, numpy as np
from scipy.optimize import differential_evolution
from motor import Motor
M = Motor(umax=0, Ts=1e-4, T=1.0)
def f(x):
    t, y, u, J = M.response(np.array(x)); return J if np.isfinite(J) else 1e6
out = {}
for B in (20, 50, 100):
    r = differential_evolution(f, [(0.001, B)]*3, seed=1, popsize=20, maxiter=300, tol=1e-12, polish=False)
    t, y, u, J = M.response(r.x)
    out[B] = dict(K=r.x.tolist(), J=float(J), Vpk=float(np.abs(u).max()))
    print(B, np.round(r.x, 4), round(J, 7), 'Vpk', round(np.abs(u).max()))
json.dump(out, open('ideal_bound.json', 'w'), indent=1)
