import ctypes, os, numpy as np
from scipy.linalg import expm
from motor import MOTOR, DEFAULT, Motor
_l = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libdcsim4.so'))
_dp = ctypes.POINTER(ctypes.c_double); D_, I_ = ctypes.c_double, ctypes.c_int
_l.sim4.restype = D_
_l.sim4.argtypes = [D_]*6 + [_dp,_dp,D_,D_,I_,I_,I_,D_,_dp,D_,I_,_dp,_dp,_dp]
def ss3(m):
    Ra,La,J,B,Kt,Kb = (m[k] for k in ('Ra','La','J','B','Kt','Kb'))
    A = np.array([[-Ra/La,-Kb/La,0],[Kt/J,-B/J,0],[0,1,0]]); Bm = np.array([[1/La,0],[0,-1/J],[0,0]]); return A,Bm
class Sim4:
    def __init__(self, motor=None, **kw):
        self.m = dict(MOTOR); self.m.update(motor or {}); self.p = dict(DEFAULT); self.p.update(kw)
        A,B = ss3(self.m); Ts = self.p['Ts']; M = np.zeros((5,5)); M[:3,:3]=A; M[:3,3:]=B; E = expm(M*Ts)
        self.Ad = np.ascontiguousarray(E[:3,:3].ravel()); self.Bd = np.ascontiguousarray(E[:3,3:].ravel())
        self.n = int(round(self.p['T']/Ts))+1
    def run(self, K, r=1.0, TL=0.0, dom=0, counts=0, noise=None, tf=0.0, delay=0, out=True, umax=None, N=None):
        p = dict(self.p); p['N'] = p['N'] if N is None else N; n = self.n; umax = p['umax'] if umax is None else umax
        beta = 1.0 if tf <= 0 else p['Ts']/(p['Ts']+tf)
        nz = None if noise is None else np.ascontiguousarray(noise, dtype=float)
        P = lambda a: None if a is None else a.ctypes.data_as(_dp)
        y = np.zeros(n) if out else None; ym = np.zeros(n) if out else None; u = np.zeros(n) if out else None
        J = _l.sim4(K[0],K[1],K[2],p['N'],p['Ts'],umax,P(self.Ad),P(self.Bd),r,TL,n,p['aw'],dom,float(counts),P(nz),beta,delay,P(y),P(ym),P(u))
        return (J, np.arange(n)*p['Ts'], y, ym, u) if out else J
