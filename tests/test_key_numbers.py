"""Regression tests for the key numbers reported in the paper. Run: python -m pytest tests (after code/build_libs.py)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from formulation import Problem
from motor import Motor
from sim4 import Sim4

_X = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'results', 'pareto_300_150_75.json')))['150.0']['x']
KSTAR, NSTAR = np.array(_X[:3]), _X[3]                           # constrained optimum, Mu_max = 150

def test_constrained_optimum():
    P = Problem(Mu_max=150.0)
    Jr, Jl = P.parts(KSTAR, NSTAR); Ms, Mu = P.margins(KSTAR, NSTAR)
    assert abs((Jr + Jl) - 5.109e-3) < 2e-6
    assert abs(Ms - 1.400) < 2e-3 and abs(Mu - 150.0) < 0.5
    assert abs(KSTAR[0] - 32.794) < 1e-3 and abs(NSTAR - 74.3) < 0.05
    assert P.stable(KSTAR, NSTAR)
    assert P.cost(np.r_[KSTAR, NSTAR]) < 5.12e-3              # penalty is negligible at the optimum

def test_idealized_bound_optimum():
    ideal = Motor(umax=0, Ts=1e-4, T=1.0)
    _, _, u, J = ideal.response(np.array([20.0, 5.3587, 3.3496]))
    assert abs(J - 4.03e-4) < 2e-6 and abs(np.abs(u).max() - 352) < 2

def test_published_mmpa_ratio():
    P = Problem(Mu_max=1e9); J = P.J(np.array([20.0, 0.7448, 1.7395]), 100.0)
    assert abs(J / 0.0051086 - 2.4) < 0.05

def test_sim4_matches_two_state_simulator():
    """The three-state simulator with an ideal sensor reproduces the two-state reference simulator."""
    K = np.array([50.803096, 5.355493, 2.162121])
    assert abs(Sim4().run(K, dom=0, N=100.0, out=False) - Motor().J_ref(K)) < 1e-12


def test_highgain_limit_of_idealized_cost():
    """Proposition 1: alpha * J_id(alpha K) tends to C(K) > 0, so J_id(alpha K) -> 0 (ideal derivative, no saturation)."""
    from scipy import signal
    K = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'results', 'ideal_bound.json')))['20']['K']
    t = np.linspace(0, 1, 200001); aJ = []
    for a in (1, 10, 100):
        Kp, Ki, Kd = [a * k for k in K]
        _, e = signal.impulse(([1.08, 6.1, 1.63], [1.08, 6.1 + 15 * Kd, 1.63 + 15 * Kp, 15 * Ki]), T=t)
        aJ.append(a * np.trapezoid(t * np.abs(e), t))
    assert abs(aJ[0] - 6.36e-4) < 1e-6 and aJ[0] > aJ[1] > aJ[2] > 2.09e-4 and abs(aJ[2] - 2.14e-4) < 2e-6
