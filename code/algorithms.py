"""Population-based optimizers used in the paper: DE/rand/1/bin, PSO (inertia 0.9 -> 0.4), GWO, HHO and a
real-coded GA with rank selection, scattered crossover and Gaussian mutation. All share an evaluation budget
through NFECounter. Signature: alg(fobj, lb, ub, N, budget, rng) -> (best_x, best_f, convergence_curve)."""
import numpy as np
from math import gamma, sin, pi

class NFECounter:
    def __init__(self, f, budget):
        self.f, self.budget, self.n = f, budget, 0
        self.bestF, self.bestX = np.inf, None
        self.curve = np.full(budget, np.nan)

    def eval(self, x):
        if self.n >= self.budget:
            return np.inf
        y = self.f(x); self.n += 1
        if y < self.bestF:
            self.bestF, self.bestX = y, np.array(x, float).copy()
        self.curve[self.n - 1] = self.bestF
        return y

    def done(self):
        return self.n >= self.budget

    def final_curve(self):
        c = self.curve.copy(); last = np.where(~np.isnan(c))[0][-1]; c[last + 1:] = c[last]
        return c

    def out(self):
        return self.bestX, self.bestF, self.final_curve()


# ----------------------------------------------------------------- DE/rand/1/bin
def de(fobj, lb, ub, N, budget, rng):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float); D = lb.size; F = NFECounter(fobj, budget)
    X = lb + (ub - lb) * rng.random((N, D)); C = np.array([F.eval(x) for x in X])
    while not F.done():
        for i in range(N):
            if F.done(): break
            idx = np.delete(np.arange(N), i); r = idx[rng.permutation(N - 1)[:3]]
            v = np.minimum(np.maximum(X[r[0]] + 0.5 * (X[r[1]] - X[r[2]]), lb), ub)
            m = rng.random(D) < 0.9; m[rng.integers(D)] = True
            u = X[i].copy(); u[m] = v[m]; fu = F.eval(u)
            if fu <= C[i]: X[i] = u; C[i] = fu
    return F.out()


# ----------------------------------------------------------------- PSO (w 0.9->0.4)
def pso(fobj, lb, ub, N, budget, rng):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float); D = lb.size; F = NFECounter(fobj, budget)
    vmax = 0.2 * (ub - lb); X = lb + (ub - lb) * rng.random((N, D)); V = np.zeros((N, D))
    C = np.array([F.eval(x) for x in X]); P = X.copy(); PC = C.copy()
    T = (budget - N) // N; t = 0
    while not F.done():
        t += 1; w = 0.9 - 0.5 * min(t / T, 1); g = P[np.argmin(PC)].copy()
        for i in range(N):
            if F.done(): break
            V[i] = np.minimum(np.maximum(w * V[i] + 2 * rng.random(D) * (P[i] - X[i]) + 2 * rng.random(D) * (g - X[i]), -vmax), vmax)
            X[i] = np.minimum(np.maximum(X[i] + V[i], lb), ub); c = F.eval(X[i])
            if c < PC[i]: P[i] = X[i]; PC[i] = c
    return F.out()


# ----------------------------------------------------------------- GWO
def gwo(fobj, lb, ub, N, budget, rng):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float); D = lb.size; F = NFECounter(fobj, budget)
    X = lb + (ub - lb) * rng.random((N, D)); C = np.array([F.eval(x) for x in X])
    T = (budget - N) // N; t = 0
    while not F.done():
        o = np.argsort(C, kind='stable'); Ld = X[o[:3]].copy(); a = 2 - 2 * min(t / T, 1); t += 1
        for i in range(N):
            if F.done(): break
            s = np.zeros(D)
            for q in range(3):
                A = 2 * a * rng.random(D) - a; Cc = 2 * rng.random(D)
                s += Ld[q] - A * np.abs(Cc * Ld[q] - X[i])
            X[i] = np.minimum(np.maximum(s / 3, lb), ub); C[i] = F.eval(X[i])
    return F.out()


# ----------------------------------------------------------------- HHO
def _levy(D, rng):
    beta = 1.5
    sig = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return 0.01 * rng.standard_normal(D) * sig / np.abs(rng.standard_normal(D)) ** (1 / beta)


def hho(fobj, lb, ub, N, budget, rng):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float); D = lb.size; F = NFECounter(fobj, budget)
    clip = lambda y: np.minimum(np.maximum(y, lb), ub)
    X = lb + (ub - lb) * rng.random((N, D)); C = np.array([F.eval(x) for x in X])
    T = int((budget - N) // (1.5 * N)); t = 0
    while not F.done():
        i0 = np.argmin(C); Cr = C[i0]; Xr = X[i0].copy(); E1 = 2 * (1 - min(t / T, 1)); t += 1
        for i in range(N):
            if F.done(): break
            E = E1 * (2 * rng.random() - 1); J = 2 * (1 - rng.random())
            if abs(E) >= 1:
                if rng.random() >= 0.5:
                    Xk = X[rng.integers(N)]; new = Xk - rng.random() * np.abs(Xk - 2 * rng.random() * X[i])
                else:
                    new = (Xr - X.mean(0)) - rng.random() * (lb + rng.random() * (ub - lb))
                X[i] = clip(new); C[i] = F.eval(X[i])
            else:
                r = rng.random()
                if r >= 0.5 and abs(E) >= 0.5:
                    X[i] = clip((Xr - X[i]) - E * np.abs(J * Xr - X[i])); C[i] = F.eval(X[i])
                elif r >= 0.5:
                    X[i] = clip(Xr - E * np.abs(Xr - X[i])); C[i] = F.eval(X[i])
                else:
                    Y = Xr - E * np.abs(J * Xr - X[i]) if abs(E) >= 0.5 else Xr - E * np.abs(J * Xr - X.mean(0))
                    Y = clip(Y); fy = F.eval(Y)
                    if fy < C[i]: X[i] = Y; C[i] = fy
                    else:
                        Z = clip(Y + rng.random(D) * _levy(D, rng)); fz = F.eval(Z)
                        if fz < C[i]: X[i] = Z; C[i] = fz
            if C[i] < Cr: Cr = C[i]; Xr = X[i].copy()
    return F.out()


# ----------------------------------------------------------------- GA (MATLAB-ga-like)
def ga(fobj, lb, ub, N, budget, rng):
    """Rank fitness scaling, stochastic-uniform selection, elite = ceil(0.05N),
    crossover fraction 0.8 (scattered), Gaussian mutation with linearly shrinking scale,
    bound clipping. Approximates MATLAB ga defaults (exact operators differ)."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float); D = lb.size; F = NFECounter(fobj, budget)
    elite = int(np.ceil(0.05 * N)); G = (budget - N) // (N - elite)
    X = lb + (ub - lb) * rng.random((N, D)); C = np.array([F.eval(x) for x in X])
    for g in range(G):
        if F.done(): break
        o = np.argsort(C, kind='stable'); X, C = X[o], C[o]
        fit = 1 / np.sqrt(np.arange(1, N + 1)); fit = fit / fit.sum()
        nkids = N - elite; nx = int(round(0.8 * nkids)); nm = nkids - nx
        cum = np.cumsum(fit); step = 1 / (2 * nx + nm); start = rng.random() * step
        ptr = start + step * np.arange(2 * nx + nm); par = np.searchsorted(cum, ptr); par = np.minimum(par, N - 1)
        par = par[rng.permutation(par.size)]
        kids = []
        for k in range(nx):
            a, b = X[par[2 * k]], X[par[2 * k + 1]]; m = rng.random(D) < 0.5; kids.append(np.where(m, a, b))
        scale = 1.0 * (1 - g / G)
        for k in range(nm):
            p = X[par[2 * nx + k]]; kids.append(np.minimum(np.maximum(p + scale * 0.1 * (ub - lb) * rng.standard_normal(D), lb), ub))
        kids = np.array(kids)
        Ck = np.array([F.eval(x) for x in kids])
        X = np.vstack([X[:elite], kids]); C = np.r_[C[:elite], Ck]
    return F.out()


ALGS = {'DE': de, 'PSO': pso, 'GA': ga, 'GWO': gwo, 'HHO': hho}
