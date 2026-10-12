import json, numpy as np
from scipy.stats import friedmanchisquare, mannwhitneyu
JSTAR = 0.0051086
for B in (2000, 5000):
    d = json.load(open(f'alg2_{B}.json'))['res']; A = list(d); F = {a: np.array(d[a]['F']) for a in A}
    print(f'== budget {B} NFE, J* = {JSTAR}')
    for a in A:
        f = F[a]; X = np.array(d[a]['X'])
        print(f"{a:4s} best {f.min():.6f} med {np.median(f):.6f} mean {f.mean():.6f} sd {f.std(ddof=1):.2e} worst {f.max():.6f} "
              f"succ1% {int((f<=1.01*JSTAR).sum()):2d}/30 succ5% {int((f<=1.05*JSTAR).sum()):2d}/30")
    st, p = friedmanchisquare(*[F[a] for a in A])
    r = np.argsort(np.argsort(np.column_stack([F[a] for a in A]), axis=1), axis=1) + 1; mr = dict(zip(A, r.mean(0)))
    print(f' Friedman chi2 = {st:.1f}, p = {p:.2e}; mean ranks', {a: round(v, 2) for a, v in sorted(mr.items(), key=lambda kv: kv[1])})
    ref = min(mr, key=mr.get); ps = {a: mannwhitneyu(F[ref], F[a], alternative='two-sided').pvalue for a in A if a != ref}
    o = sorted(ps, key=ps.get); m = len(o); adj = {}; run = 0
    for i, a in enumerate(o): run = max(run, min(1, (m-i)*ps[a])); adj[a] = run
    print(f' Mann-Whitney vs {ref} (Holm):', {a: f'{adj[a]:.1e}' for a in A if a != ref})
    # where do failures end?
    for a in A:
        X = np.array(d[a]['X']); bad = F[a] > 1.05*JSTAR
        if bad.any(): print(f'   {a} failures (J>1.05J*): Ki range {X[bad,1].min():.1f}-{X[bad,1].max():.1f}, N range {X[bad,3].min():.0f}-{X[bad,3].max():.0f}, J {F[a][bad].min():.4f}-{F[a][bad].max():.4f}')
