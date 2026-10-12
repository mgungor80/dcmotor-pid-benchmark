"""Convergence summary from alg2_<budget>.json written by run_algs2.py (with per-run 'curve'):
median and quartiles of the best-so-far cost, and evaluations needed to come within 1% of J*."""
import sys, json, numpy as np
JSTAR = 0.0051086; B = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
d = json.load(open(f'alg2_{B}.json'))['res']; out = {}
for a, v in d.items():
    C = np.array(v['curve']) / JSTAR; hit = [int(np.argmax(c <= 1.01)) + 1 for c in C if (c <= 1.01).any()]
    out[a] = dict(median=np.median(C, 0).round(6).tolist(), q1=np.percentile(C, 25, 0).round(6).tolist(), q3=np.percentile(C, 75, 0).round(6).tolist(),
                  evals_to_1pct=hit, median_evals_to_1pct=float(np.median(hit)) if hit else None, median_at_1000=float(np.median(C[:, 999])))
    print(a, 'successful', len(hit), 'median evals to 1%', out[a]['median_evals_to_1pct'], 'median J/J* at 1000', round(out[a]['median_at_1000'], 3))
json.dump(out, open(f'alg2_{B}_curves.json', 'w'))
