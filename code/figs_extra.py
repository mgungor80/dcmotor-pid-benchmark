"""Figures 1, 4 and 8 of the article: closed-loop block diagram, derivative time against realistic cost, optimizer convergence.
Run from results/ (needs lit_table.json and alg2_5000_curves.json)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.stats import spearmanr
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8, 'axes.labelsize': 8,
    'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'axes.grid': True, 'grid.color': '#d9d9d6', 'grid.linewidth': 0.4, 'axes.edgecolor': '#52514e',
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, GRAY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#52514e', '#0b0b0b'
JSTAR = 0.0051086
def save(fig, name):
    fig.savefig(name + '.png'); fig.savefig(name + '.pdf'); plt.close(fig)

# ---------------- block diagram
fig, a = plt.subplots(figsize=(7.0, 2.35)); a.set_xlim(0, 100); a.set_ylim(0, 33); a.axis('off')
def box(x, y, w, h, txt, fc='#f1f5fb', ec=BLUE, fs=7):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=0.8', fc=fc, ec=ec, lw=0.9))
    a.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=INK)
def arrow(x0, y0, x1, y1, txt=None, dx=0, dy=1.2, ha='center'):
    a.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK, shrinkA=0, shrinkB=0, mutation_scale=7))
    if txt: a.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, txt, ha=ha, va='bottom', fontsize=7, color=INK)
def summ(x, y, signs):
    a.add_patch(Circle((x, y), 1.6, fc='white', ec=INK, lw=0.8))
    for s, (ox, oy) in signs: a.text(x + ox, y + oy, s, fontsize=7, ha='center', va='center', color=INK)
Y = 22
a.text(0.5, Y, '$r$', fontsize=8, va='center'); arrow(2.2, Y, 8.4, Y)
summ(10, Y, [('+', (-2.6, 1.6)), ('−', (1.0, -3.0))])
arrow(11.6, Y, 17.0, Y, '$e_k$')
box(17.5, Y - 4.2, 15, 8.4, 'P + I\n$K_p e_k + I_k$\n(clamping)')
arrow(33.3, Y, 37.6, Y)
summ(39.2, Y, [('+', (-2.6, 1.6)), ('+', (1.1, -3.0))])
arrow(40.8, Y, 45.0, Y, '$v_k$')
box(45.5, Y - 4.2, 10, 8.4, 'saturation\n±12 V', fc='#fff4ee', ec=ORANGE)
arrow(55.9, Y, 60.0, Y, '$u_k$')
box(60.5, Y - 4.2, 15.5, 8.4, 'ZOH, $T_s$ = 1 ms\nDC motor $G(s)$', fc='#eefaf5', ec=AQUA)
a.text(68.2, Y + 9.4, '$T_L$', fontsize=8, ha='center'); arrow(68.2, Y + 8.6, 68.2, Y + 4.6)
arrow(76.4, Y, 81.6, Y, '$\\omega$')
box(82, Y - 4.2, 15.5, 8.4, 'sensor: encoder\nquantization, noise', fc='#f6f4fe', ec=VIOLET)
# feedback path (bottom) with a branch to the derivative block
FB = 3.5
a.plot([89.8, 89.8, 10, 10], [Y - 4.6, FB, FB, Y - 4], color=INK, lw=0.8)
a.annotate('', xy=(10, Y - 1.7), xytext=(10, Y - 4), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK, mutation_scale=7))
a.text(91, 9.5, '$y_k$', fontsize=8)
a.add_patch(Circle((39.2, FB), 0.5, fc=INK, ec=INK))
box(28.2, 8.0, 22, 6.6, 'derivative on measurement\n$D_k=-K_d\\,N\\,(y_k-y_{k-1})/(1+NT_s)$', fs=6.5)
a.annotate('', xy=(39.2, 7.6), xytext=(39.2, FB), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK, mutation_scale=7))
a.annotate('', xy=(39.2, Y - 1.7), xytext=(39.2, 15.0), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK, mutation_scale=7))
a.text(40.2, 16.0, '$D_k$', fontsize=7.5)
fig.tight_layout(pad=0.2); save(fig, 'Fig1_block_diagram')

# ---------------- derivative time vs realistic cost
L = json.load(open('lit_table.json'))
hi = [r for r in L if r['K'][0] >= 10]; lo = [r for r in L if r['K'][0] < 10]
rho, p = spearmanr([r['K'][2] / r['K'][0] for r in hi], [r['ratio'] for r in hi])
fig, a = plt.subplots(figsize=(3.4, 2.6))
a.plot([r['K'][2] / r['K'][0] for r in hi], [r['ratio'] for r in hi], 'o', color=BLUE, ms=4.5, label=r'published, $K_p\geq 10$')
a.plot([r['K'][2] / r['K'][0] for r in lo], [r['ratio'] for r in lo], 's', color=ORANGE, mfc='white', ms=4.5, mew=1.0, label=r'published, $K_p<10$')
a.plot(1.6370 / 32.794, 1.0, '*', color=INK, ms=9, label='This work (nominal)')
lab = {'MMPA': (5, -2), 'KOA': (6, 3), 'RPOA': (20, -4), 'ASO': (-14, 14), 'ASO (quoted in [Ayinla])': (-8, 4), 'BKA-ANN': (20, -17), 'SFS': (5, -2), 'PSO': (5, -2), 'IWO': (5, -2), 'SCA': (-5, 0), 'GWO': (-5, -4)}
nice = {'ASO (quoted in [Ayinla])': 'ASO, quoted in [10]'}
for r in L:
    m = {'ASO-q': 'ASO (quoted in [Ayinla])'}.get(r.get('label'), r.get('label', r.get('method'))); xy = (r['K'][2] / r['K'][0], r['ratio'])
    if m in lab:
        a.annotate(nice.get(m, m), xy, xytext=lab[m], textcoords='offset points', fontsize=6, color=GRAY, va='center' if m in ('SCA', 'GWO') else 'baseline', ha='right' if m in ('SCA', 'GWO', 'ASO') else 'left',
                    arrowprops=dict(arrowstyle='-', lw=0.4, color=GRAY, shrinkA=1, shrinkB=3) if m in ('RPOA', 'ASO', 'BKA-ANN') else None)
    if m == 'LHHO':
        a.annotate('LHHO and four sets\nwith similar gains', xy, xytext=(4, -26), textcoords='offset points', fontsize=6, color=GRAY,
                   arrowprops=dict(arrowstyle='-', lw=0.5, color=GRAY, shrinkA=0, shrinkB=3))
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(r'Derivative time $T_d=K_d/K_p$ (s)'); a.set_ylabel(r'Realistic cost $J/J^\star$')
a.text(0.03, 0.97, f'high-gain sets: Spearman $\\rho$ = {rho:.2f}, $p$ = {p*1e4:.1f}$\\times10^{{-4}}$', transform=a.transAxes, fontsize=6.5, va='top', color=INK)
a.legend(frameon=False, loc='lower right', fontsize=6); a.set_ylim(0.7, 60)
fig.tight_layout(pad=0.3); save(fig, 'Fig4_derivative_time')
print('Td spearman', rho, p)

# ---------------- convergence
cols = {'DE': BLUE, 'PSO': ORANGE, 'GA': AQUA, 'GWO': VIOLET, 'HHO': MAGENTA}; ls = {'DE': '-', 'PSO': '--', 'GA': '-.', 'GWO': ':', 'HHO': (0, (5, 1, 1, 1))}
d = json.load(open('alg2_5000_curves.json'))
fig, a = plt.subplots(figsize=(3.4, 2.6)); x = np.arange(1, 5001)
for k in ('DE', 'PSO', 'GA', 'GWO', 'HHO'):
    a.plot(x, d[k]['median'], ls=ls[k], color=cols[k], label=k); a.fill_between(x, d[k]['q1'], d[k]['q3'], color=cols[k], alpha=0.12, lw=0)
a.axhline(1.01, color=GRAY, lw=0.6, ls='--'); a.text(33, 1.03, '1 % above $J^\\star$', fontsize=6, color=GRAY, ha='left', va='bottom')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(30, 5000); a.set_ylim(0.95, 30)
a.set_xlabel('Cost evaluations'); a.set_ylabel(r'Best cost so far $J/J^\star$')
a.legend(frameon=False, ncol=2, loc='upper right', fontsize=6)
fig.tight_layout(pad=0.3); save(fig, 'Fig8_convergence')
print('ok')
