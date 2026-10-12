#!/usr/bin/env bash
# Reproduce the tables and figures of the paper.
#   ./reproduce.sh          quick: recompute every table and figure from the stored optimization results (minutes)
#   ./reproduce.sh full     also re-run all optimizations (several hours on a laptop)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; C="$ROOT/code"
python "$C/build_libs.py"
cd "$ROOT/results"
if [ "${1:-quick}" = "full" ]; then
  python "$C/ideal_bound.py"                                   # Section 2.3: idealized optimum vs box size
  python "$C/noiseaware.py" 1000 200 100; python "$C/noiseaware.py" 300 150 75
  python "$C/noiseaware.py" 50 25 10;     python "$C/noiseaware.py" 35 15     # Mu trade-off (Fig. 5a)
  python "$C/robustdesign.py" 1000 150;   python "$C/robustdesign.py" 50 25   # scenario-robust designs (Table 2)
  python "$C/vsens2.py" 12,1 6,1 24,1 6,0.5 12,0.5 24,0.5
  python "$C/vsens2.py" 6,2 12,2 24,2 6,5 12,5 24,5             # supply voltage x sampling period (Table 5)
  python "$C/run_algs2.py" 2000; python "$C/run_algs2.py" 5000   # optimizer comparison, well-posed problem (Table 4, Figs. 7-8)
  python "$C/conv_summary.py" 2000; python "$C/conv_summary.py" 5000   # convergence summaries (Fig. 8)
  python "$C/ideal_algs.py" 2000; python "$C/ideal_algs.py" 5000 # optimizer comparison, idealized problem (Table 4)
fi
python "$C/highgain_limit.py" > highgain_limit.txt             # Proposition 1, Section 2.3
python "$C/lit_table.py"                                       # Table 1, Fig. 2
python "$C/evalpareto.py"    > pareto_eval.txt                 # Fig. 5, Table S1
python "$C/montecarlo.py" 500 > mc.txt                         # Fig. 6b, Table S2
python "$C/mc_summary.py"    > mc_summary.txt
python "$C/eval_robust.py"   > robust_eval.txt                 # Table 2, Fig. 6a
python "$C/vsens_lit.py"     > vsens2_table.txt                # Table 5
python "$C/stats2.py"        > stats2.txt                      # Table 4 (well-posed problem)
python "$C/ideal_stats.py"   > ideal_stats.txt                 # Table 4 (idealized problem)
python "$C/si_tables.py"     > si_tables.txt                   # Table 3
python "$C/step_amplitude.py" > step_amplitude.txt             # Section 5.3: reference amplitude
python "$C/reproduce_published.py" > reproduce_published.txt   # Section 2.5: published results in their own setting
python "$C/figs_main.py"  && python "$C/figs_extra.py"         # Figures 1-8
mv -f Fig[1-8]_* "$ROOT/figures/"
echo "done: see results/ and figures/"
