"""Test robust vs nominal designs on the 500 test plants (set B, seed 7) and in the sensor scenarios."""
import json, glob, numpy as np
from motor import MOTOR, metrics
from sim4 import Sim4
from montecarlo import Gz, Cz, cl_stable
from noiseaware import margins as nom_margins, Jnom
KEYS = ('Ra','La','J','B','Kt','Kb'); F = np.random.default_rng(7).uniform(0.8, 1.2, (500, 6))
SIMS = []
for f in F:
    m = dict(MOTOR); m.update({k: MOTOR[k]*fi for k, fi in zip(KEYS, f)}); s = Sim4(motor=m); SIMS.append((s, Gz(s)))
nomP = json.load(open('pareto_eval.json'))['K']; rob = {}
for f in glob.glob('robust_[0-9]*.json'): rob.update(json.load(open(f)))
C = []
for mu in sorted(rob, key=float, reverse=True):
    C.append((f'Nominal Mu<={float(mu):g}', nomP[f'Mu<={float(mu):g}']))
    C.append((f'Robust  Mu<={float(mu):g}', (rob[mu]['x'][:3], rob[mu]['x'][3])))
SC = [('sig=0.02',0,0.02),('enc 2^16',65536,0),('enc 2^14',16384,0)]
S0 = Sim4(); n = S0.n; out = {}
print(f"{'design':20s} {'Kp':>7s} {'Ki':>7s} {'Kd':>6s} {'N':>6s} {'Jnom':>8s} {'Msnom':>6s} | {'J med':>8s} {'J p95':>8s} {'Ms p95':>6s} {'Ms max':>6s} {'Ms>1.4':>6s} | " + ' '.join(f'{s[0]:>9s}' for s in SC))
for name, (K, N) in C:
    Ms0, Mu0 = nom_margins(np.array(K), N); J0 = Jnom(K, N); R = []
    for s, G in SIMS:
        Ms = float(np.max(np.abs(1/(1+Cz(K,N)*G)))); J = s.run(K, dom=1, N=N, out=False) + s.run(K, r=0, TL=1e-3, dom=1, N=N, out=False)
        R.append((J, Ms, cl_stable(s, K, N)))
    R = np.array(R, float); sens = []
    for sn, cnt, sig in SC:
        Js = []
        for sd in range(20 if sig > 0 else 1):
            nz = np.random.default_rng(sd).normal(0, sig, n) if sig > 0 else None; nz2 = np.random.default_rng(500+sd).normal(0, sig, n) if sig > 0 else None
            Js.append(S0.run(K, dom=1, N=N, counts=cnt, noise=nz, out=False) + S0.run(K, r=0, TL=1e-3, dom=1, N=N, counts=cnt, noise=nz2, out=False))
        sens.append(float(np.mean(Js)))
    out[name] = dict(K=list(K), N=N, Jnom=J0, Msnom=Ms0, Munom=Mu0, J=R[:,0].tolist(), Ms=R[:,1].tolist(), stable=float(R[:,2].mean()), sens=sens)
    print(f"{name:20s} {K[0]:7.3f} {K[1]:7.4f} {K[2]:6.4f} {N:6.1f} {J0:8.5f} {Ms0:6.3f} | {np.median(R[:,0]):8.5f} {np.percentile(R[:,0],95):8.5f} "
          f"{np.percentile(R[:,1],95):6.3f} {R[:,1].max():6.3f} {100*(R[:,1]>1.4).mean():5.1f}% | " + ' '.join(f'{v:9.5f}' for v in sens) + f"  stable {100*R[:,2].mean():.0f}%")
json.dump(dict(scen=[s[0] for s in SC], res=out), open('robust_eval.json', 'w'))
