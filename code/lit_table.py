"""Table 1 and Figure 1: published PID gains for the benchmark motor, evaluated
(a) in the idealized setting (no saturation, derivative on the error with N = 100, Ts = 0.1 ms, ITAE over 1 s) and
(b) in the formulation of the paper (derivative on the measurement, N = 100, Ts = 1 ms, +-12 V, clamping): J, Ms, Mu.
Run from the results directory; reads ../data/published_gains.csv, writes lit_table.json/.csv/.txt."""
import csv, json, os, numpy as np
from motor import Motor
from formulation import Problem
HERE = os.path.dirname(os.path.abspath(__file__))
GAINS = os.path.join(HERE, '..', 'data', 'published_gains.csv')
JSTAR = 0.0051086                      # constrained optimum for Mu_max = 150 (noiseaware.py, pareto_300_150_75.json)
def main():
    ideal = Motor(umax=0, Ts=1e-4, T=1.0); P = Problem(Mu_max=1e9); rows = []
    hdr = f"{'label':9s} {'Kp':>8s} {'Ki':>8s} {'Kd':>7s} | {'ITAEid':>8s} {'reported':>8s} {'Vpk':>6s} | {'J':>8s} {'J/J*':>5s} {'Ms':>5s} {'Mu':>6s}"
    lines = [hdr]
    for g in csv.DictReader(open(GAINS, encoding='utf-8')):
        K = np.array([float(g['Kp']), float(g['Ki']), float(g['Kd'])]); t, y, u, Ji = ideal.response(K)
        Ms, Mu = P.margins(K, 100.0); Jr = P.J(K, 100.0)
        ub = float(g['box_upper']) if g['box_upper'] else None; rep = float(g['reported_cost']) if g['reported_cost'] else None
        rows.append(dict(label=g['label'], method=g['method'], source=g['source'], quoted_in=g['quoted_in'], K=K.tolist(),
                         ITAE_ideal=Ji, reported=rep, Vpeak=float(abs(u).max()), J_real=Jr, ratio=Jr / JSTAR, Ms=Ms, Mu=Mu,
                         at_bound=bool(ub and K.max() >= 0.98 * ub)))
        lines.append(f"{g['label']:9s} {K[0]:8.4f} {K[1]:8.4f} {K[2]:7.4f} | {Ji:8.6f} {('' if rep is None else f'{rep:.6f}'):>8s} {abs(u).max():6.0f} | "
                     f"{Jr:8.5f} {Jr / JSTAR:5.1f} {Ms:5.3f} {Mu:6.0f}")
    json.dump(rows, open('lit_table.json', 'w'), indent=1)
    with open('lit_table.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['label', 'Kp', 'Ki', 'Kd', 'ITAE_ideal', 'reported', 'Vpeak', 'J_real', 'ratio', 'Ms', 'Mu', 'at_bound'])
        for r in rows: w.writerow([r['label'], *r['K'], r['ITAE_ideal'], r['reported'], r['Vpeak'], r['J_real'], r['ratio'], r['Ms'], r['Mu'], r['at_bound']])
    open('lit_table.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
if __name__ == '__main__':
    main()
