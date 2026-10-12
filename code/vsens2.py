"""Vmax x Ts sensitivity of the journal formulation (Mu <= 150, Ms <= 1.4), re-optimised per cell,
plus the 12 V / 1 ms design applied without retuning, and the best published controller (N = 100)."""
import sys, json, numpy as np
from scipy.optimize import differential_evolution
from formulation import Problem, BOUNDS
from motor import metrics
# Published gain sets (see data/published_gains.csv)
LIT = {'LHHO':(19.99999,5.437263,3.359391),'BKA-ANN':(24.9369,2.9429,4.9641),'MMPA':(20.0,0.7448,1.7395),
       'KOA':(17.6949,5.1085,4.6981),'RPOA':(18.3100,4.9058,4.0094),'LFDNM':(19.5593,5.1566,3.4097),'ASO':(11.9437,2.0521,2.4358),
       'GWO':(6.8984,0.5626,0.9293),'SCA':(4.5012,0.5260,0.5302),'SFS':(1.6315,0.2798,0.2395),'PSO':(1.5234,1.3801,0.0159),'IWO':(1.5782,0.4372,0.0481)}
X0 = [32.794, 3.6596, 1.6370, 74.3]   # will be replaced by the 12 V / 1 ms result
def solve(P):
    best = None
    for seed in (1, 2):
        r = differential_evolution(P.cost, BOUNDS, seed=seed, popsize=25, maxiter=600, tol=1e-12, polish=False, mutation=(0.5,1), recombination=0.9)
        if best is None or r.fun < best.fun: best = r
    return best.x, best.fun
def describe(P, x):
    K, N = np.array(x[:3]), x[3]; Jr, Jl = P.parts(K, N); Ms, Mu = P.margins(K, N)
    J, t, y, ym, u = P.sim.run(K, dom=1, N=N); mt = metrics(t, y)
    sat = np.abs(u) >= P.umax-1e-9; tsat = t[np.where(sat)[0][-1]]*1e3 if sat.any() else 0.0
    return dict(K=K.tolist(), N=float(N), J=Jr+Jl, Jr=Jr, Jl=Jl, Ms=Ms, Mu=Mu, OS=mt['OS'], ts=mt['ts']*1e3, tsat=tsat, stable=bool(P.stable(K, N)))
if __name__ == '__main__':
    cells = [(float(a.split(',')[0]), float(a.split(',')[1])) for a in sys.argv[1:]]
    out = {}
    for V, Ts in cells:
        P = Problem(Ts=Ts*1e-3, umax=V); x, f = solve(P); d = describe(P, x)
        lit = {n: P.J(np.array(K), 100.0) for n, K in LIT.items() if P.stable(np.array(K), 100.0)}
        bl = min(lit, key=lit.get); d['best_lit'] = (bl, lit[bl], P.margins(np.array(LIT[bl]), 100.0)[0]); d['penalised'] = f
        out[f'{V:g},{Ts:g}'] = d
        print(f"V={V:4g} Ts={Ts:3g}ms K=({d['K'][0]:.3f},{d['K'][1]:.4f},{d['K'][2]:.4f}) N={d['N']:.1f} J={d['J']:.6f} "
              f"(Jr {d['Jr']:.6f} Jl {d['Jl']:.6f}) Ms={d['Ms']:.3f} Mu={d['Mu']:.1f} OS={d['OS']:.2f}% ts={d['ts']:.0f}ms tsat={d['tsat']:.0f}ms "
              f"| best lit {bl} J={lit[bl]:.6f} ratio {lit[bl]/d['J']:.1f}", flush=True)
    json.dump(out, open('vsens2_%s.json' % '_'.join(a.replace(',', 'V').replace('.', 'p') for a in sys.argv[1:]), 'w'), indent=1)
