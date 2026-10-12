import json, numpy as np
d = json.load(open('mc_results.json')); ev = json.load(open('pareto_eval.json'))
Jn = {r[0]: r[3] for r in ev['rows']}
for s in ('A', 'B'):
    print(f"== set {s}: {d['keys'][s]} +-20 %, {d['NS']} samples")
    print(f"{'controller':18s} {'Jnom':>8s} {'J med':>8s} {'J p95':>8s} {'p95/nom':>7s} {'Ms max':>6s} {'OS p95':>6s} {'ts p95':>6s} {'unsettled':>9s}")
    for n, R in d['res'][s].items():
        R = np.array(R, float); ok = R[:, 6] == 1; J = R[ok, 0]; ts = R[ok, 5]; fin = np.isfinite(ts)
        tsp = np.percentile(np.where(fin, ts, 9.99), 95)
        print(f"{n:18s} {Jn[n]:8.5f} {np.median(J):8.5f} {np.percentile(J,95):8.5f} {np.percentile(J,95)/Jn[n]:7.2f} "
              f"{R[ok,3].max():6.3f} {np.percentile(R[ok,4],95):6.2f} {('>1 s' if tsp>1 else f'{tsp*1e3:4.0f}ms'):>6s} {100*(~fin).mean():8.1f}%")
