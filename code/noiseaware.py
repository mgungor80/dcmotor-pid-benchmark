"""Noise-aware robustness-constrained formulation: x = (Kp, Ki, Kd, N).
min J = ITAE_ref + ITAE_load (nominal, D on measurement)  s.t. Ms <= 1.4, Mu = max|C S| <= Mu_max."""
import numpy as np, json, sys
from motor import Motor
from sim4 import Sim4
from scipy.optimize import differential_evolution
M = Motor(); S = Sim4(); z = M._z; G = M.Gz; Ts = 1e-3
Ad = M._Ad.reshape(2,2); bu = M._Bd.reshape(2,2)[:,0]
def Cz(K, N):
    Kp,Ki,Kd = K; return Kp + Ki*Ts*z/(z-1) + Kd*N*(1-1/z)/(1+N*Ts-1/z)
def stable(K, N):
    Kp,Ki,Kd = K; a = 1/(1+N*Ts)
    def step(s):
        x=s[:2]; I,D,ep=s[2],s[3],s[4]; e=-x[1]; In=I+Ki*Ts*e; Dn=a*(D+Kd*N*(e-ep)); u=Kp*e+In+Dn
        return np.r_[Ad@x+bu*u, In, Dn, e]
    Mx = np.column_stack([step(c) for c in np.eye(5)]); return np.max(np.abs(np.linalg.eigvals(Mx))) < 1-1e-9
def margins(K, N):
    C = Cz(K,N); Sn = 1/(1+C*G); return float(np.max(np.abs(Sn))), float(np.max(np.abs(C*Sn)))
def Jnom(K, N, dom=1):
    return S.run(K, dom=dom, N=N, out=False) + S.run(K, r=0, TL=1e-3, dom=dom, N=N, out=False)
def cost(x, Mu_max, Ms_max=1.4, w=10.0):
    K, N = x[:3], x[3]
    if not stable(K, N): return 1e6
    Ms, Mu = margins(K, N); J = Jnom(K, N)
    return J + w*max(0, Ms-Ms_max) + w*max(0, Mu/Mu_max-1)
if __name__ == '__main__':
    out = {}
    for Mu_max in [float(a) for a in sys.argv[1:]]:
        best = None
        for seed in (1, 2, 3):
            r = differential_evolution(cost, [(0,100),(0,100),(0,10),(5,500)], args=(Mu_max,), seed=seed,
                                       popsize=25, maxiter=600, tol=1e-12, polish=False, mutation=(0.5,1), recombination=0.9)
            if best is None or r.fun < best.fun: best = r
        K, N = best.x[:3], best.x[3]; Ms, Mu = margins(K, N)
        out[Mu_max] = dict(x=best.x.tolist(), J=best.fun, Jnom=Jnom(K,N), Ms=Ms, Mu=Mu)
        print(f'Mu_max={Mu_max:6.1f}  K=({K[0]:.3f},{K[1]:.4f},{K[2]:.4f}) N={N:.1f}  J={best.fun:.6f} Ms={Ms:.3f} Mu={Mu:.1f}', flush=True)
    json.dump(out, open('pareto_%s.json' % '_'.join(sys.argv[1:]), 'w'))
