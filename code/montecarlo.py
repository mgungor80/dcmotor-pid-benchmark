"""Parametric uncertainty Monte Carlo: motor parameters uniform +-20 % (set A: Ra, J, B as planned;
set B: all six Ra, La, J, B, Kt, Kb). Ideal sensor, D on measured speed, Ts = 1 ms, +-12 V, clamping.
Common random samples for all controllers."""
import json, sys, numpy as np
from motor import MOTOR, metrics
from sim4 import Sim4
Ts = 1e-3; w = np.logspace(-2, np.log10(np.pi/Ts*0.999), 3000); z = np.exp(1j*w*Ts)
def Gz(sim):
    Ad = sim.Ad.reshape(3,3)[:2,:2]; b = sim.Bd.reshape(3,2)[:2,0]
    a,bb,c,d = Ad[0,0],Ad[0,1],Ad[1,0],Ad[1,1]; det = (z-a)*(z-d)-bb*c
    return (c*b[0] + (z-a)*b[1])/det          # second row of (zI-Ad)^-1 b
def Cz(K,N): return K[0] + K[1]*Ts*z/(z-1) + K[2]*N*(1-1/z)/(1+N*Ts-1/z)
def cl_stable(sim, K, N):
    Ad = sim.Ad.reshape(3,3)[:2,:2]; bu = sim.Bd.reshape(3,2)[:2,0]; a = 1/(1+N*Ts)
    def step(s):
        x=s[:2]; I,D,yp=s[2],s[3],s[4]; y=x[1]; e=-y; In=I+K[1]*Ts*e; Dn=a*(D-K[2]*N*(y-yp)); u=K[0]*e+In+Dn
        return np.r_[Ad@x+bu*u, In, Dn, y]
    M = np.column_stack([step(c) for c in np.eye(5)]); return np.max(np.abs(np.linalg.eigvals(M))) < 1-1e-9
if __name__ == "__main__":
    d = json.load(open('pareto_eval.json')); CTRL = d['K']
    KEYS = {'A': ('Ra','J','B'), 'B': ('Ra','La','J','B','Kt','Kb')}
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 500; res = {}
    for setn, keys in KEYS.items():
        rng = np.random.default_rng(7); F = rng.uniform(0.8, 1.2, (NS, len(keys)))
        sims = []
        for f in F:
            m = dict(MOTOR); m.update({k: MOTOR[k]*fi for k, fi in zip(keys, f)}); s = Sim4(motor=m); sims.append((s, Gz(s)))
        res[setn] = {}
        for name, (K, N) in CTRL.items():
            R = []
            for s, G in sims:
                if not cl_stable(s, K, N): R.append((np.nan,)*6 + (0,)); continue
                Ms = float(np.max(np.abs(1/(1+Cz(K,N)*G))))
                J, t, y, ym, u = s.run(K, dom=1, N=N); Jl = s.run(K, r=0, TL=1e-3, dom=1, N=N, out=False)
                mt = metrics(t, y); R.append((J+Jl, J, Jl, Ms, mt['OS'], mt['ts'], 1))
            R = np.array(R, float); res[setn][name] = R.tolist()
            ok = R[:,6] == 1; Jt = R[ok,0]
            print(f"{setn} {name:18s} stable {ok.mean()*100:5.1f}%  J med {np.median(Jt):.5f} p95 {np.percentile(Jt,95):.5f} max {Jt.max():.5f} "
                  f"(J/Jnom p95 {np.percentile(Jt,95)/R[0,0] if False else 0:.0f})  Ms med {np.median(R[ok,3]):.3f} max {R[ok,3].max():.3f}  "
                  f"OS p95 {np.percentile(R[ok,4],95):5.2f}%  ts p95 {np.percentile(R[ok,5],95)*1e3:5.0f} ms", flush=True)
        res[setn+'_F'] = F.tolist()
    json.dump(dict(keys=KEYS, NS=NS, res=res), open('mc_results.json', 'w'))
