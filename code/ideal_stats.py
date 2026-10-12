import json, numpy as np
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata
for B in (2000, 5000):
    d = json.load(open(f'ideal_algs_{B}.json')); A = list(d['res']); F = {a: np.array(d['res'][a]['F']) for a in A}; J0 = d['Jref']
    st, p = friedmanchisquare(*[F[a] for a in A]); mr = dict(zip(A, np.mean([rankdata(r) for r in np.column_stack([F[a] for a in A])], 0)))
    ref = min(mr, key=mr.get); ps = {a: mannwhitneyu(F[ref], F[a], alternative='two-sided').pvalue for a in A if a != ref}
    o = sorted(ps, key=ps.get); m = len(o); adj = {}; run = 0
    for i, a in enumerate(o): run = max(run, min(1, (m - i) * ps[a])); adj[a] = run
    print(B, f'chi2 {st:.1f} p {p:.1e}')
    for a in A: print(f'  {a:4s} best {F[a].min()/J0:.4f} med {np.median(F[a])/J0:.4f} succ {(F[a] <= 1.01*J0).sum()} rank {mr[a]:.2f} p {adj.get(a, float("nan")):.1e} worst {F[a].max()/J0:.3f}')
