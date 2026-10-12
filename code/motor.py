"""Benchmark DC motor (Ra=0.4, La=2.7, J=4e-4, B=2.2e-3, Kt=0.015, Kb=0.05):
w/V = 15/(1.08 s^2 + 6.1 s + 1.63).  Realistic formulation: digital PID (Ts = 1 ms,
N = 100), +-12 V saturation with clamping, true ITAE, load-torque case, Ms constraint."""
import ctypes, os, numpy as np
from scipy.linalg import expm
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libdcsim2.so'))
_dp = ctypes.POINTER(ctypes.c_double)
_lib.sim2.restype = ctypes.c_double
_lib.sim2.argtypes = [ctypes.c_double]*6 + [_dp, _dp, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, _dp, _dp]

MOTOR = dict(Ra=0.4, La=2.7, J=4e-4, B=2.2e-3, Kt=0.015, Kb=0.05)
DEFAULT = dict(N=100.0, Ts=1e-3, T=1.0, umax=12.0, aw=1, r=1.0, TL=1e-3, Ms_max=1.4, w_pen=10.0, w_load=1.0)

def motor_ss(m=MOTOR):
    Ra, La, J, B, Kt, Kb = (m[k] for k in ('Ra', 'La', 'J', 'B', 'Kt', 'Kb'))
    A = np.array([[-Ra/La, -Kb/La], [Kt/J, -B/J]])
    Bm = np.array([[1/La, 0], [0, -1/J]])           # inputs: V, TL
    return A, Bm

def zoh(A, B, Ts):
    n, m = B.shape; M = np.zeros((n+m, n+m)); M[:n, :n] = A; M[:n, n:] = B
    E = expm(M*Ts); return E[:n, :n], E[:n, n:]

class Motor:
    def __init__(self, motor=None, **kw):
        self.m = dict(MOTOR); self.m.update(motor or {})
        self.p = dict(DEFAULT); self.p.update(kw)
        A, B = motor_ss(self.m); Ts = self.p['Ts']
        Ad, Bd = zoh(A, B, Ts)
        self._Ad = np.ascontiguousarray(Ad.ravel()); self._Bd = np.ascontiguousarray(Bd.ravel())
        self.n = int(round(self.p['T']/Ts)) + 1
        self.w = np.logspace(-2, np.log10(np.pi/Ts*0.999), 3000); z = np.exp(1j*self.w*Ts); self._z = z
        bu = Bd[:, 0]
        self.Gz = np.array([np.linalg.solve(zz*np.eye(2)-Ad, bu)[1] for zz in z])
    def _p(self, a): return a.ctypes.data_as(_dp)
    def itae(self, K, r, TL, umax=None):
        p = self.p; umax = p['umax'] if umax is None else umax
        return _lib.sim2(K[0], K[1], K[2], p['N'], p['Ts'], umax, self._p(self._Ad), self._p(self._Bd),
                         r, TL, self.n, p['aw'], None, None)
    def J_ref(self, K): return self.itae(K, self.p['r'], 0.0)
    def J_load(self, K): return self.itae(K, 0.0, self.p['TL'])
    def Cz(self, K):
        Kp, Ki, Kd = K; Ts, N = self.p['Ts'], self.p['N']; z = self._z
        return Kp + Ki*Ts*z/(z-1) + Kd*N*(1-1/z)/(1+N*Ts-1/z)
    def Ms(self, K): return float(np.max(np.abs(1/(1+self.Cz(K)*self.Gz))))
    def stable(self, K):
        """unsaturated closed loop: Nyquist-free check via closed-loop eigenvalues"""
        Ad = self._Ad.reshape(2, 2); Bd = self._Bd.reshape(2, 2)[:, 0]
        Kp, Ki, Kd = K; Ts, N = self.p['Ts'], self.p['N']; a = 1/(1+N*Ts)
        # states [x(2), I, D, eprev] with r=0, e=-w ; u = Kp e + I(k) + D(k) where I,D updated with e(k)
        # build linear map numerically by applying the update to unit vectors
        def step(s):
            x = s[:2]; I, D, ep = s[2], s[3], s[4]; e = -x[1]
            In = I + Ki*Ts*e; Dn = a*(D + Kd*N*(e-ep)); u = Kp*e + In + Dn
            xn = Ad @ x + Bd*u; return np.r_[xn, In, Dn, e]
        M = np.column_stack([step(c) for c in np.eye(5)])
        return np.max(np.abs(np.linalg.eigvals(M))) < 1 - 1e-9
    def J(self, K, two=True):
        """design objective: ITAE_ref + w_load*ITAE_load + penalty on Ms > Ms_max"""
        p = self.p
        if not self.stable(K): return 1e6
        J = self.J_ref(K) + (p['w_load']*self.J_load(K) if two else 0.0)
        v = self.Ms(K) - p['Ms_max']
        return J + (p['w_pen']*v if v > 0 else 0.0)
    def response(self, K, r=None, TL=0.0, umax=None, T=None):
        p = self.p; r = p['r'] if r is None else r; umax = p['umax'] if umax is None else umax
        n = self.n if T is None else int(round(T/p['Ts'])) + 1
        y = np.zeros(n); u = np.zeros(n)
        J = _lib.sim2(K[0], K[1], K[2], p['N'], p['Ts'], umax, self._p(self._Ad), self._p(self._Bd), r, TL, n, p['aw'],
                      self._p(y), self._p(u))
        return np.arange(n)*p['Ts'], y, u, J

def metrics(t, y, r=1.0, band=0.02):
    os_ = max(0.0, (y.max()-r)/r*100); i10 = np.argmax(y >= 0.1*r); i90 = np.argmax(y >= 0.9*r)
    out = np.where(np.abs(y-r) > band*abs(r))[0]
    ts = 0.0 if not len(out) else (t[out[-1]+1] if out[-1]+1 < len(t) else np.inf)
    return dict(OS=os_, tr=t[i90]-t[i10] if y.max() >= 0.9*r else np.inf, ts=ts, ess=abs(r-y[-1]))

# ---- derivative-on-measurement variant (same feedback loop, so same Ms)
_lib3 = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libdcsim3.so'))
_lib3.sim3.restype = ctypes.c_double
_lib3.sim3.argtypes = [ctypes.c_double]*6 + [_dp, _dp, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, _dp, _dp]

class MotorDoM(Motor):
    """PID with derivative acting on the measured speed (-Kd*N*s/(s+N) * w)."""
    def itae(self, K, r, TL, umax=None):
        p = self.p; umax = p['umax'] if umax is None else umax
        return _lib3.sim3(K[0], K[1], K[2], p['N'], p['Ts'], umax, self._p(self._Ad), self._p(self._Bd),
                          r, TL, self.n, p['aw'], 1, None, None)
    def response(self, K, r=None, TL=0.0, umax=None, T=None):
        p = self.p; r = p['r'] if r is None else r; umax = p['umax'] if umax is None else umax
        n = self.n if T is None else int(round(T/p['Ts'])) + 1
        y = np.zeros(n); u = np.zeros(n)
        J = _lib3.sim3(K[0], K[1], K[2], p['N'], p['Ts'], umax, self._p(self._Ad), self._p(self._Bd), r, TL, n, p['aw'], 1,
                       self._p(y), self._p(u))
        return np.arange(n)*p['Ts'], y, u, J
