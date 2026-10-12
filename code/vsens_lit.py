"""Table 5: for each supply voltage and sampling period, the re-optimized design (read from vsens2_*.json),
the 12 V / 1 ms design used without re-tuning, and the best published gain set."""
import json, glob, numpy as np
from formulation import Problem
from vsens2 import LIT
D = {}
for f in glob.glob('vsens2_*V*.json'): D.update(json.load(open(f)))
x0 = D['12,1']; K0, N0 = np.array(x0['K']), x0['N']
rows = []
print(f"{'Vmax':>4s} {'Ts':>4s} | {'Kp':>7s} {'Ki':>6s} {'Kd':>6s} {'N':>6s} {'J':>9s} {'OS%':>5s} {'ts ms':>5s} {'tsat':>4s} | {'J fixed':>8s} {'Ms fix':>6s} {'loss%':>6s} | best lit (ratio)")
for key in sorted(D, key=lambda k: (float(k.split(',')[0]), float(k.split(',')[1]))):
    V, Ts = map(float, key.split(',')); d = D[key]; P = Problem(Ts=Ts*1e-3, umax=V)
    lit = {n: P.J(np.array(K), 100.0) for n, K in LIT.items() if P.stable(np.array(K), 100.0)}
    bl = min(lit, key=lit.get)
    Jf = P.J(K0, N0) if P.stable(K0, N0) else np.nan; Msf = P.margins(K0, N0)[0]
    rows.append(dict(V=V, Ts=Ts, K=d['K'], N=d['N'], J=d['J'], OS=d['OS'], ts=d['ts'], tsat=d['tsat'], Jfix=Jf, Msfix=Msf, best_lit=bl, J_lit=lit[bl]))
    print(f"{V:4g} {Ts:4g} | {d['K'][0]:7.3f} {d['K'][1]:6.3f} {d['K'][2]:6.3f} {d['N']:6.1f} {d['J']:9.6f} {d['OS']:5.2f} {d['ts']:5.0f} {d['tsat']:4.0f} | "
          f"{Jf:8.6f} {Msf:6.3f} {100*(Jf/d['J']-1):6.1f} | {bl} {lit[bl]:.5f} ({lit[bl]/d['J']:.1f}x)")
json.dump(rows, open('vsens_lit.json', 'w'), indent=1)
