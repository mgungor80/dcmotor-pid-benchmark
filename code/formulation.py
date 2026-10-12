"""Journal formulation (single source of truth).
x = (Kp, Ki, Kd, N); digital PID, derivative on measured speed, Ts, +-umax, clamping anti-windup.
J = ITAE(speed step r=1) + ITAE(load step TL=1e-3)        (nominal plant, ideal sensor)
s.t. Ms <= Ms_max, Mu = max|C S| <= Mu_max, closed loop stable.  Static penalty w=10."""
import numpy as np
from motor import MOTOR, DEFAULT
from sim4 import Sim4
BOUNDS = [(0, 100), (0, 100), (0, 10), (5, 500)]
class Problem:
    def __init__(self, Ts=1e-3, umax=12.0, Mu_max=150.0, Ms_max=1.4, motor=None, w=10.0, T=1.0):
        self.Ts, self.umax, self.Mu_max, self.Ms_max, self.w = Ts, umax, Mu_max, Ms_max, w
        self.sim = Sim4(motor=motor, Ts=Ts, umax=umax, T=T)
        self.wgrid = np.logspace(-2, np.log10(np.pi/Ts*0.999), 3000); self.z = np.exp(1j*self.wgrid*Ts)
        Ad = self.sim.Ad.reshape(3,3)[:2,:2]; b = self.sim.Bd.reshape(3,2)[:2,0]; self.Ad2, self.bu = Ad, b
        a,bb,c,d = Ad[0,0],Ad[0,1],Ad[1,0],Ad[1,1]; z = self.z; det = (z-a)*(z-d)-bb*c
        self.G = (c*b[0] + (z-a)*b[1])/det
    def C(self, K, N):
        z, Ts = self.z, self.Ts; return K[0] + K[1]*Ts*z/(z-1) + K[2]*N*(1-1/z)/(1+N*Ts-1/z)
    def margins(self, K, N):
        C = self.C(K, N); S = 1/(1+C*self.G); return float(np.abs(S).max()), float(np.abs(C*S).max())
    def stable(self, K, N):
        Ts = self.Ts; a = 1/(1+N*Ts); Ad, bu = self.Ad2, self.bu
        def step(s):
            x=s[:2]; I,D,yp=s[2],s[3],s[4]; y=x[1]; e=-y; In=I+K[1]*Ts*e; Dn=a*(D-K[2]*N*(y-yp)); u=K[0]*e+In+Dn
            return np.r_[Ad@x+bu*u, In, Dn, y]
        M = np.column_stack([step(c) for c in np.eye(5)]); return np.max(np.abs(np.linalg.eigvals(M))) < 1-1e-9
    def J(self, K, N):
        s = self.sim; return s.run(K, dom=1, N=N, out=False) + s.run(K, r=0, TL=1e-3, dom=1, N=N, out=False)
    def parts(self, K, N):
        s = self.sim; return s.run(K, dom=1, N=N, out=False), s.run(K, r=0, TL=1e-3, dom=1, N=N, out=False)
    def cost(self, x):
        x = np.asarray(x, float); K, N = x[:3], x[3]
        if not self.stable(K, N): return 1e6
        Ms, Mu = self.margins(K, N)
        pen = self.w*max(0, Ms-self.Ms_max) + self.w*max(0, Mu/self.Mu_max-1)
        return self.J(K, N) + pen
    __call__ = cost
