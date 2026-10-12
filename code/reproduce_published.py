"""Section 2.5: published controllers simulated in their own setting (continuous time, no saturation,
derivative on the error), either ideal or with a first-order derivative filter N = 100 (the default of the
Simulink PID block): rise time (10-90 %), 2 % settling time, overshoot and ITAE, for comparison with the
values reported in the original studies."""
import numpy as np
from scipy import signal
G_NUM, G_DEN = np.array([15.0]), np.array([1.08, 6.1, 1.63])     # G(s) = 15 / (1.08 s^2 + 6.1 s + 1.63)
def closed_loop(K, N=None):
    Kp, Ki, Kd = K
    if N is None: cn, cd = np.array([Kd, Kp, Ki]), np.array([1.0, 0.0])            # Kp + Ki/s + Kd s
    else: cn, cd = np.polyadd(np.polymul([Kp, Ki], [1, N]), [Kd * N, 0, 0]), np.array([1.0, N, 0.0])
    L_n, L_d = np.polymul(cn, G_NUM), np.polymul(cd, G_DEN)
    return L_n, np.polyadd(L_d, L_n)
def metrics(K, N=None, T=1.0):
    num, den = closed_loop(K, N); t = np.linspace(0, T, int(T * 200000) + 1); _, y = signal.step(signal.lti(num, den), T=t)
    e = np.abs(1 - y); out = np.where(e > 0.02)[0]
    return dict(tr=t[np.argmax(y >= 0.9)] - t[np.argmax(y >= 0.1)], ts=t[out[-1]] if len(out) else 0.0,
                OS=max(0.0, (y.max() - 1) * 100), ITAE=np.trapezoid(t * e, t))
CASES = {'KOA (Saravanan 2025)': (17.6949, 5.1085, 4.6981), 'RPOA (Saravanan 2025)': (18.3100, 4.9058, 4.0094),
         'SFS (Khanam and Parmar 2017)': (1.6315, 0.2798, 0.2395), 'LHHO (Ayinla 2024)': (19.99999, 5.437263, 3.359391)}
for N in (None, 100.0):
    print('ideal derivative' if N is None else f'derivative filter N = {N:g}')
    for name, K in CASES.items():
        m = metrics(K, N); m05 = metrics(K, N, 0.5)
        print(f"  {name:30s} tr {m['tr']:.4f} s  ts {m['ts']:.4f} s  OS {m['OS']:.2f} %  ITAE(1 s) {m['ITAE']:.6f}  ITAE(0.5 s) {m05['ITAE']:.6f}")
