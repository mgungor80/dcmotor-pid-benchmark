"""Table 3 (derivative realization) and supplemental Table S2.
S1: cost ratio of the published gain sets for four derivative realizations.
S5: Monte Carlo summary (500 test plants, all six parameters +-20 %) of nominal designs and published gains.
Run from the results directory after lit_table.py, evalpareto.py and montecarlo.py."""
import json, numpy as np
from sim4 import Sim4
JSTAR = 0.0051086
S = Sim4(); out = []
print('S1  label      meas N100  err N100  meas N1000  err N1000')
for r in json.load(open('lit_table.json')):
    K = np.array(r['K']); v = {}
    for dom in (1, 0):
        for N in (100.0, 1000.0):
            v[f'{dom}_{int(N)}'] = (S.run(K, dom=dom, N=N, out=False) + S.run(K, r=0, TL=1e-3, dom=dom, N=N, out=False)) / JSTAR
    out.append(dict(label=r['label'], v=v)); print(f"    {r['label']:9s} {v['1_100']:9.1f} {v['0_100']:9.1f} {v['1_1000']:11.1f} {v['0_1000']:10.1f}")
json.dump(out, open('si_derivative.json', 'w'), indent=1)
d = json.load(open('mc_results.json'))['res']['B']; Jn = {x[0]: x[3] for x in json.load(open('pareto_eval.json'))['rows']}; mc = []
print('S5  design               Jnom    median  p95     Ms max  Ms>1.4 %')
for n, Rr in d.items():
    A = np.array(Rr, float); ok = A[:, 6] == 1; J = A[ok, 0]
    m = dict(name=n, Jnom=Jn[n], med=float(np.median(J)), p95=float(np.percentile(J, 95)), msmax=float(A[ok, 3].max()),
             ms14=float(100 * np.mean(A[ok, 3] > 1.4)), stable=float(100 * ok.mean())); mc.append(m)
    print(f"    {n:20s} {m['Jnom']*1e3:6.2f} {m['med']*1e3:7.2f} {m['p95']*1e3:7.2f} {m['msmax']:7.3f} {m['ms14']:7.1f}")
json.dump(mc, open('si_mc.json', 'w'), indent=1)
