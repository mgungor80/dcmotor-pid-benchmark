"""Proposition 1: idealized ITAE (ideal derivative, no saturation, 1 s) of the scaled bound optimum alpha*K, and the limit C(K)."""
import json, numpy as np
from scipy import signal
K = json.load(open('ideal_bound.json'))['20']['K']; t = np.linspace(0, 1, 400001); out = {}
for a in (1, 2, 5, 10, 20, 50, 100):
    Kp, Ki, Kd = [a * k for k in K]
    _, e = signal.impulse(([1.08, 6.1, 1.63], [1.08, 6.1 + 15 * Kd, 1.63 + 15 * Kp, 15 * Ki]), T=t)   # error of a unit step
    out[a] = float(np.trapezoid(t * np.abs(e), t)); print(a, f'{out[a]:.4g}', f'alpha*J = {a * out[a]:.4g}')
Kp, Ki, Kd = K; Q = np.array([Kd, Kp, Ki]); num = np.array([1.08, 6.1, 1.63]) - (1.08 / Kd) * Q
_, eps = signal.impulse((num[1:], 15 * Q), T=t); C = float(np.trapezoid(t * np.abs(eps), t))
print('C(K) =', f'{C:.4g}', ' PID zeros', np.roots(Q))
json.dump(dict(K=K, J=out, C=C, zeros=np.roots(Q).tolist()), open('highgain_limit.json', 'w'), indent=1)
