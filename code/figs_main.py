"""Figures 1-5 of the paper (PNG at 600 dpi and PDF). Run from the results directory."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from motor import Motor
from sim4 import Sim4

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8, 'axes.labelsize': 8,
    'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'axes.grid': True, 'grid.color': '#d9d9d6', 'grid.linewidth': 0.4, 'axes.edgecolor': '#52514e',
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
BLUE, ORANGE, AQUA, VIOLET, GRAY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#52514e', '#0b0b0b'
JSTAR = 0.0051086
OPT = (np.array([32.794, 3.6596, 1.6370]), 74.3)          # nominal, Mu <= 150
ROB = json.load(open('robust_eval.json'))['res']
ROBK = (np.array(ROB['Robust  Mu<=150']['K']), ROB['Robust  Mu<=150']['N'])
def save(fig, name):
    fig.savefig(name + '.png'); fig.savefig(name + '.pdf'); plt.close(fig)

# ---------------- Fig. 1: published gains, idealised ITAE vs realistic cost
L = json.load(open('lit_table.json'))
ideal = Motor(umax=0, Ts=1e-4, T=1.0, N=OPT[1]); _, _, _, Jid_opt = ideal.response(OPT[0])
fig, a = plt.subplots(figsize=(3.4, 2.6))
for r in L:
    hi = r['K'][0] >= 10
    a.plot(r['ITAE_ideal'], r['ratio'], 'o' if hi else 's', ms=4.5, color=BLUE if hi else ORANGE,
           mfc=BLUE if hi else 'white', mew=1.0)
lab = {'LHHO': ('LHHO and four sets with similar gains', (-8, -13)), 'MMPA': ('MMPA', (5, -2)), 'BKA-ANN': ('BKA-ANN', (-22, -11)),
       'KOA': ('KOA', (4, 3)), 'RPOA': ('RPOA', (-12, 6)), 'SFS': ('SFS', (-18, 4)), 'PSO': ('PSO', (-8, 6)), 'IWO': ('IWO', (5, -5)),
       'GWO': ('GWO', (4, -8)), 'SCA': ('SCA', (4, 3)), 'ASO': ('ASO', (5, 2)), 'ASO-q': ('ASO, quoted in [10]', (-7, -2))}
for r in L:
    if r['label'] in lab:
        txt, off = lab[r['label']]
        a.annotate(txt, (r['ITAE_ideal'], r['ratio']), xytext=off, textcoords='offset points', fontsize=6, color=GRAY,
                   ha='right' if r['label'] == 'ASO-q' else 'left', va='center' if r['label'] == 'ASO-q' else 'baseline')
a.plot(Jid_opt, 1.0, '*', ms=9, color=INK)
a.annotate('This work', (Jid_opt, 1.0), xytext=(6, -2), textcoords='offset points', fontsize=6.5, color=INK)
a.plot([], [], 'o', ms=4.5, color=BLUE, label=r'high gain, $K_p\geq 10$')
a.plot([], [], 's', ms=4.5, color=ORANGE, mfc='white', label=r'low gain, $K_p<10$')
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel(r'Idealized ITAE $J_{id}$ (no saturation, 1 s)'); a.set_ylabel(r'Realistic cost $J/J^\star$')
a.set_ylim(0.7, 60); a.legend(frameon=False, loc='lower right', fontsize=6)
fig.tight_layout(pad=0.3); save(fig, 'Fig2_published_gains')
print('Fig1: this work idealised ITAE', round(Jid_opt, 6))

# ---------------- Fig. 2: time responses in the realistic setting
S = Sim4()
CTR = [('This work, nominal', OPT, BLUE, '-'), ('This work, scenario-robust', ROBK, AQUA, '-.'),
       ('LHHO [10]', ((19.99999, 5.437263, 3.359391), 100.0), ORANGE, '--'),
       ('MMPA [15]', ((20.0, 0.7448, 1.7395), 100.0), VIOLET, ':')]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
for name, (K, N), c, ls in CTR:
    K = np.array(K, float)
    J, t, y, ym, u = S.run(K, dom=1, N=N)
    Jl, tl, yl, yml, ul = S.run(K, r=0, TL=1e-3, dom=1, N=N)
    m = t <= 0.6
    ax[0].plot(t[m], y[m], ls, color=c, label=name); ax[1].plot(t[m], u[m], ls, color=c)
    ax[2].plot(tl[m], yl[m] * 1e3, ls, color=c)
ax[0].set_xlabel('Time (s)'); ax[0].set_ylabel('Speed (rad/s)'); ax[0].set_title('(a) reference step', fontsize=8, loc='left')
ax[1].set_xlabel('Time (s)'); ax[1].set_ylabel('Armature voltage (V)'); ax[1].set_title('(b) control signal', fontsize=8, loc='left')
ax[1].axhline(12, color=GRAY, lw=0.6, ls='--'); ax[1].axhline(-12, color=GRAY, lw=0.6, ls='--'); ax[1].set_ylim(-13, 13)
ax[2].set_xlabel('Time (s)'); ax[2].set_ylabel(r'Speed deviation ($10^{-3}$ rad/s)'); ax[2].set_title(r'(c) load step $T_L$', fontsize=8, loc='left')
ax[0].legend(frameon=False, loc='lower right', fontsize=5.8)
fig.tight_layout(pad=0.3); save(fig, 'Fig3_time_responses')

# ---------------- Fig. 3: noise sensitivity trade-off
d = json.load(open('pareto_eval.json')); R = d['rows']; sc = d['scen']
par = sorted([r for r in R if r[0].startswith('Mu<=')], key=lambda r: r[2])
lit = [r for r in R if not r[0].startswith(('Mu<=', 'Prev'))]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
a = ax[0]; mu = [r[2] for r in par]; J = [r[3] for r in par]
a.plot(mu, J, '-o', color=BLUE, ms=3.5, label=r'This work: optimum for each $M_u$ bound')
off = {'LHHO': (4, -9), 'BKA-ANN': (-20, 5), 'MMPA': (4, -3), 'ASO': (-10, 5), 'GWO': (4, -9), 'SCA': (-10, 5), 'ZN': (5, -3)}
for r in lit:
    a.plot(r[2], r[3], '^', color=GRAY, ms=4.5)
    a.annotate(r[0], (r[2], r[3]), xytext=off.get(r[0], (3, 3)), textcoords='offset points', fontsize=6, color=GRAY)
a.plot([], [], '^', color=GRAY, ms=4.5, label=r'Published gains ($N$ = 100)')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(r'Noise-to-control gain $M_u$ (V per rad/s)')
a.set_ylabel(r'Nominal cost $J$'); a.set_ylim(top=1.2); a.legend(frameon=False, loc='upper right'); a.set_title('(a)', fontsize=8, loc='left')
b = ax[1]; cols = [BLUE, ORANGE, AQUA, VIOLET]; mk = ['o', 's', 'D', '^']; ls = ['-', '--', '-.', ':']
labn = {'sig=0.005': r'white noise $\sigma$ = 0.005 rad/s', 'sig=0.02': r'white noise $\sigma$ = 0.02 rad/s',
        'enc 2^16': r'encoder $2^{16}$ counts/rev', 'enc 2^14': r'encoder $2^{14}$ counts/rev'}
for i, s in enumerate(sc):
    Js = [r[4 + 2 * i] for r in par]; b.plot(mu, Js, ls=ls[i], marker=mk[i], ms=3, color=cols[i], label=labn[s])
    k = int(np.argmin(Js)); b.plot(mu[k], Js[k], marker=mk[i], ms=7, mfc='none', mec=cols[i], mew=1.0)
b.set_xscale('log'); b.set_yscale('log'); b.set_xlabel(r'$M_u$ of the tuned controller (V per rad/s)')
b.set_ylabel(r'Cost $J$ with sensor imperfection'); b.legend(frameon=False, loc='upper center', ncol=2, fontsize=6)
b.set_ylim(top=0.6); b.set_title('(b)', fontsize=8, loc='left')
fig.tight_layout(pad=0.4); save(fig, 'Fig5_noise_tradeoff')

# ---------------- Fig. 4: parametric uncertainty
mc = json.load(open('mc_results.json'))['res']['B']; mus = ['1000', '150', '50', '25']
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
a = ax[0]; pos = np.arange(len(mus))
for o, kind, c in ((-0.18, 'Nominal', ORANGE), (0.18, 'Robust ', BLUE)):
    data = [ROB[f'{kind} Mu<={m}']['Ms'] for m in mus]
    a.boxplot(data, positions=pos + o, widths=0.3, whis=(5, 95), showfliers=False, patch_artist=True,
              medianprops=dict(color='white', lw=1.2), boxprops=dict(facecolor=c, edgecolor=c),
              whiskerprops=dict(color=c), capprops=dict(color=c))
    a.plot(pos + o, [max(v) for v in data], 'x', color=c, ms=4)
a.axhline(1.4, color=GRAY, lw=0.8, ls='--'); a.set_xlim(-0.5, 4.35)
a.text(4.32, 1.396, r'$M_s$ = 1.4', fontsize=6, ha='right', va='top', color=GRAY)
a.set_xticks(pos); a.set_xticklabels([m for m in mus]); a.set_xlabel(r'$M_u$ bound (V per rad/s)', labelpad=1); a.set_ylabel(r'$M_s$ over 500 perturbed plants')
a.plot([], [], 's', color=ORANGE, label='Nominal design'); a.plot([], [], 's', color=BLUE, label='Scenario-robust design')
a.plot([], [], 'x', color=GRAY, label='maximum'); a.legend(frameon=False, loc='lower center', ncol=3, columnspacing=1.0, handletextpad=0.3, fontsize=6)
a.set_ylim(1.15, 1.62); a.set_title('(a)', fontsize=8, loc='left')
b = ax[1]; x = [float(m) for m in mus]
for kind, c, mkr in (('Nominal', ORANGE, 's'), ('Robust ', BLUE, 'o')):
    med = [np.median(ROB[f'{kind} Mu<={m}']['J']) for m in mus]; p95 = [np.percentile(ROB[f'{kind} Mu<={m}']['J'], 95) for m in mus]
    b.plot(x, med, '-', marker=mkr, color=c, ms=3.5, label=f'{kind.strip()}: median')
    b.plot(x, p95, ':', marker=mkr, color=c, ms=3.5, mfc='white', label=f'{kind.strip()}: 95th pct.')
mud = {r[0]: r[2] for r in R}
for n in ('LHHO', 'BKA-ANN', 'MMPA', 'ASO', 'GWO', 'SCA'):
    p = np.percentile(np.array(mc[n])[:, 0], 95); b.plot(mud[n], p, '^', color=GRAY, ms=4.5)
    b.annotate(n, (mud[n], p), xytext={'LHHO': (4, -8), 'BKA-ANN': (-20, 5), 'ASO': (-10, 5)}.get(n, (3, 3)),
               textcoords='offset points', fontsize=6, color=GRAY)
b.plot([], [], '^', color=GRAY, label='Published: 95th pct.')
b.set_xscale('log'); b.set_yscale('log'); b.set_xlabel(r'$M_u$ bound (V per rad/s)', labelpad=1); b.set_ylabel('Cost $J$ over 500 perturbed plants')
b.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6, columnspacing=0.8); b.set_title('(b)', fontsize=8, loc='left')
fig.tight_layout(pad=0.4); save(fig, 'Fig6_uncertainty')

# ---------------- Fig. 5: optimizer comparison
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for a, B in zip(ax, (2000, 5000)):
    res = json.load(open(f'alg2_{B}.json'))['res']; A = list(res)
    data = [np.array(res[k]['F']) / JSTAR for k in A]
    a.boxplot(data, widths=0.5, whis=(0, 100), showfliers=False, medianprops=dict(color=BLUE, lw=1.4),
              boxprops=dict(color=GRAY), whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY))
    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        a.plot(i + 1 + rng.uniform(-0.15, 0.15, len(v)), v, 'o', ms=2.2, color=BLUE, alpha=0.55, mew=0)
    a.axhline(1.05, color=GRAY, lw=0.6, ls='--')
    a.set_xticks(range(1, len(A) + 1)); a.set_xticklabels(A); a.set_yscale('log')
    a.set_title(f'({"ab"[B == 5000]}) {B} evaluations, 30 runs', fontsize=8, loc='left')
ax[0].set_ylabel(r'Final cost $J/J^\star$'); ax[1].text(0.52, 1.06, '+5 %', fontsize=6, color=GRAY, ha='left', va='bottom')
fig.tight_layout(pad=0.4); save(fig, 'Fig7_optimizers')
print('ok')
