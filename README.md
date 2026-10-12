# A well-posed DC motor PID tuning benchmark

Code and data for the article

> M. Güngör, *A well-posed formulation of the DC motor PID tuning benchmark: Saturation, robustness, and noise constraints change the ranking of controllers and optimizers*, submitted to the International Journal of Control (2026).

A second-order DC motor model, G(s) = 15 / (1.08 s² + 6.1 s + 1.63), is widely used to compare metaheuristic algorithms for PID tuning. In its usual form the controller is continuous, the amplifier is unbounded and the cost is the ITAE of a reference step; this cost has no minimizer (scaling any positive gain vector up drives it to zero), so the reported optimum lies on the bound of the search box. This repository provides a well-posed version of the benchmark, the published gain sets used in the article, and scripts that regenerate every table and figure.

## The benchmark

| Item | Definition |
|---|---|
| Motor | R<sub>a</sub> = 0.4 Ω, L<sub>a</sub> = 2.7 H, J<sub>m</sub> = 4×10⁻⁴ kg m², B = 2.2×10⁻³ N m s/rad, K<sub>t</sub> = 0.015 N m/A, K<sub>b</sub> = 0.05 V s/rad; exact zero-order-hold discretization |
| Controller | sampled PID, T<sub>s</sub> = 1 ms; derivative on the measured speed with first-order filter (constant N); integrator clamping anti-windup; amplifier ±12 V |
| Decision vector | x = (K<sub>p</sub>, K<sub>i</sub>, K<sub>d</sub>, N) in [0, 100] × [0, 100] × [0, 10] × [5, 500] |
| Cost | J = ∫₀¹ t·\|r − ω<sub>r</sub>(t)\| dt + ∫₀¹ t·\|ω<sub>L</sub>(t)\| dt, reference step r = 1 rad/s and load step T<sub>L</sub> = 10⁻³ N m, trapezoidal rule on the true speed |
| Constraints | M<sub>s</sub> = max\|S\| ≤ 1.4; M<sub>u</sub> = max\|C S\| ≤ M<sub>u</sub><sup>max</sup> (main design: 150 V per rad/s); closed-loop stability |
| Constraint handling | static penalty 10 × violation; unstable candidates cost 10⁶ |
| Reference optimum (M<sub>u</sub><sup>max</sup> = 150) | K<sub>p</sub> = 32.794, K<sub>i</sub> = 3.6596, K<sub>d</sub> = 1.6370, N = 74.26, **J\* = 5.1086×10⁻³** |

## Quick start

Requirements: Python ≥ 3.10 with NumPy, SciPy and Matplotlib, and a C compiler (gcc or clang).

```bash
pip install -r requirements.txt
python code/build_libs.py          # compiles the C simulators
./reproduce.sh                     # recomputes all tables and figures from stored results (about 1 min)
./reproduce.sh full                # also re-runs every optimization (several hours)
```

On Windows, use WSL, or compile with MinGW-w64 and adapt the library names loaded in `code/motor.py` and `code/sim4.py`.

### Evaluating your own optimizer

```python
import sys; sys.path.insert(0, 'code')
import numpy as np
from formulation import Problem, BOUNDS

P = Problem(Mu_max=150.0)                      # 12 V, Ts = 1 ms, Ms <= 1.4
x = np.array([32.794, 3.6596, 1.6370, 74.26])  # (Kp, Ki, Kd, N)
print(P.cost(x))                               # penalized cost, the objective to minimize
K, N = x[:3], x[3]
print(P.J(K, N), P.margins(K, N))              # cost, (Ms, Mu)
```

Please report J/J\*, M<sub>s</sub>, M<sub>u</sub>, the number of cost evaluations, and whether the solution lies on a bound of the box. `code/algorithms.py` shows the evaluation-budget convention used in the article (`NFECounter`).

## Repository layout

```
code/      simulators (C), formulation, optimizers, analysis and figure scripts
data/      published_gains.csv: the 17 published gain sets with their sources
results/   stored optimization results and all derived tables (JSON / TXT / CSV)
figures/   Figures 1-8 of the article (PNG and PDF)
tests/     regression tests for the key numbers
```

## From the article to the code

| Article | Script (run from `results/`) | Output |
|---|---|---|
| Section 2.3, Proposition 1: high-gain limit of the idealized cost | `highgain_limit.py` | `highgain_limit.*` |
| Section 2.3: idealized optimum vs box size | `ideal_bound.py` | `ideal_bound.json` |
| Section 2.5: published results in their own setting | `reproduce_published.py` | `reproduce_published.txt` |
| Figure 1: closed loop; Figure 4: derivative time; Figure 8: convergence | `figs_extra.py` | `figures/Fig1_*`, `Fig4_*`, `Fig8_*` |
| Table 1, Figure 2: published gains in both settings | `lit_table.py` | `lit_table.*` |
| Figure 3: time responses | `figs_main.py` | `figures/Fig3_*` |
| Table 2, Figure 6: parametric uncertainty, scenario-robust design; Table S2 | `montecarlo.py`, `robustdesign.py`, `eval_robust.py`, `mc_summary.py` | `mc_results.json`, `robust_*.json`, `robust_eval.*`, `mc_summary.txt` |
| Table 3: derivative realization | `si_tables.py` | `si_derivative.json` |
| Figure 5, Table S1: noise trade-off and sensor scenarios | `noiseaware.py`, `evalpareto.py` | `pareto_*.json`, `pareto_eval.*` |
| Table 4, Figures 7-8: optimizer comparison, well-posed problem | `run_algs2.py`, `stats2.py`, `conv_summary.py` | `alg2_*.json`, `stats2.txt`, `alg2_*_curves.json` |
| Table 4: optimizer comparison, idealized problem | `ideal_algs.py`, `ideal_stats.py` | `ideal_algs_*.json`, `ideal_stats.txt` |
| Table 5: supply voltage and sampling period | `vsens2.py`, `vsens_lit.py` | `vsens2_*.json`, `vsens2_table.txt` |
| Section 5.3: reference step amplitude | `step_amplitude.py` | `step_amplitude.*` |

The stored optimizer results (`alg2_*.json`) keep the final points and costs of every run; the per-run convergence curves written by `run_algs2.py` are summarized in `alg2_*_curves.json` (median, quartiles and evaluations needed to come within 1% of J\*).

## Published gain sets

`data/published_gains.csv` lists each gain set with its original source and, where the gains were taken from a comparison table in a later study, that study. Gain sets marked as quoted were not re-derived; please consult the original articles. Corrections are welcome through an issue.

## License

Code: MIT (see `LICENSE`). Data and figures: CC BY 4.0.

## Citation

See `CITATION.cff`. The reference will be updated when the article is published.
