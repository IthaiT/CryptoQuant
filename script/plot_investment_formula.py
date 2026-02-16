"""
Fixed-income DCA (Dollar-Cost Averaging) formula visualization.
Formula: n = ln(C(2+i) / (Bi + C(1+i))) / ln(1+i)
n: periods (months), B: principal, C: monthly investment, i: monthly rate
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "investment_formula")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Core calculations ---------------------------------------------
def calc_n_years_vec(B, C, i):
    """Vectorized: return years needed to reach target."""
    B, C, i = (np.asarray(x, float) for x in (B, C, i))
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = C * (2 + i) / (B * i + C * (1 + i))
        n = np.where((arg > 0) & (i > 0), np.log(arg) / np.log(1 + i), np.nan)
    return n / 12


def calc_n_months(B, C, i):
    """Scalar: return months needed to reach target."""
    if i <= 0:
        return np.nan
    arg = C * (2 + i) / (B * i + C * (1 + i))
    return np.log(arg) / np.log(1 + i) if arg > 0 else np.nan


# -- Plot helpers ---------------------------------------------------
def _grid(ax, xmaj, xmin, ymaj, ymin):
    for axis, maj, mn in [(ax.xaxis, xmaj, xmin), (ax.yaxis, ymaj, ymin)]:
        axis.set_major_locator(ticker.MultipleLocator(maj))
        axis.set_minor_locator(ticker.MultipleLocator(mn))
    ax.grid(True, which='major', ls='-', alpha=0.4)
    ax.grid(True, which='minor', ls=':', alpha=0.2)


def _year_refs(ax, years=(5, 10, 15, 20, 30)):
    for y in years:
        if y <= ax.get_ylim()[1]:
            ax.axhline(y=y, color='red', ls=':', alpha=0.25, lw=0.8)
            ax.text(ax.get_xlim()[1] * 0.98, y + 0.3, f'{y}yr',
                    ha='right', va='bottom', fontsize=9, color='red', alpha=0.6)


def _auto_ylim(ax, data_list, pct=95, cap=80):
    vals = np.concatenate([d[np.isfinite(d)] for d in data_list])
    if len(vals):
        ax.set_ylim(0, min(np.percentile(vals, pct), cap))


def _save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _top_year_axis(ax, max_year=30, step=5):
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = np.arange(0, max_year + 1, step)
    ax2.set_xticks(ticks * 12)
    ax2.set_xticklabels([f'{y}yr' for y in ticks])
    ax2.set_xlabel('Years', fontsize=12)


# ==================================================================
#  Chart 1: Fixed C & i, n vs B (principal)
# ==================================================================
def plot_n_vs_B():
    fig, ax = plt.subplots(figsize=(16, 10))
    B_arr = np.linspace(0, 200_000, 2000)
    rates, Cs = [0.05, 0.10, 0.15, 0.20], [2000, 5000, 8000]
    colors, styles = plt.cm.tab10.colors, ['-', '--', '-.']

    data = []
    for ci, C in enumerate(Cs):
        for ri, r in enumerate(rates):
            ny = calc_n_years_vec(B_arr, C, r / 12)
            ax.plot(B_arr / 1e4, ny, color=colors[ri], ls=styles[ci],
                    lw=1.8, label=f"C={C}/mo, rate={r*100:.0f}%", alpha=0.85)
            data.append(ny)

    ax.set_xlabel("Principal B (x10k)", fontsize=14)
    ax.set_ylabel("Years needed n", fontsize=14)
    ax.set_title("Fixed DCA amount C & rate i -- Principal B vs Years n",
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, 20)
    _grid(ax, 2, 0.5, 5, 1)
    _auto_ylim(ax, data)
    _year_refs(ax)
    ax.legend(fontsize=8, ncol=3, loc='upper right', framealpha=0.9)
    _save(fig, "01_n_vs_principal_B.png")


# ==================================================================
#  Chart 2: Fixed B & i, n vs C (monthly investment)
# ==================================================================
def plot_n_vs_C():
    fig, ax = plt.subplots(figsize=(16, 10))
    C_arr = np.linspace(1000, 10000, 2000)
    Bs, rates = [0, 50_000, 100_000, 200_000], [0.05, 0.10, 0.15, 0.20]
    colors, styles = plt.cm.tab10.colors, ['-', '--', '-.', ':']

    data = []
    for bi, B in enumerate(Bs):
        for ri, r in enumerate(rates):
            ny = calc_n_years_vec(B, C_arr, r / 12)
            ax.plot(C_arr / 1000, ny, color=colors[ri], ls=styles[bi],
                    lw=1.8, label=f"B={B//10000}w, {r*100:.0f}%", alpha=0.85)
            data.append(ny)

    ax.set_xlabel("Monthly DCA amount C (x1k)", fontsize=14)
    ax.set_ylabel("Years needed n", fontsize=14)
    ax.set_title("Fixed principal B & rate i -- DCA amount C vs Years n",
                 fontsize=16, fontweight='bold')
    ax.set_xlim(1, 10)
    _grid(ax, 1, 0.25, 5, 1)
    _auto_ylim(ax, data)
    _year_refs(ax)
    ax.legend(fontsize=8, ncol=4, loc='upper right', framealpha=0.9)
    _save(fig, "02_n_vs_monthly_C.png")


# ==================================================================
#  Chart 3: Fixed B & C, n vs i (annual rate)
# ==================================================================
def plot_n_vs_i():
    fig, ax = plt.subplots(figsize=(16, 10))
    r_arr = np.linspace(0.03, 0.25, 2000)
    Bs, Cs = [0, 50_000, 100_000, 200_000], [1000, 3000, 5000, 8000, 10000]
    colors, styles = plt.cm.tab10.colors, ['-', '--', '-.', ':']

    data = []
    for bi, B in enumerate(Bs):
        for ci, C in enumerate(Cs):
            ny = calc_n_years_vec(B, C, r_arr / 12)
            ax.plot(r_arr * 100, ny, color=colors[ci], ls=styles[bi],
                    lw=1.8, label=f"B={B//10000}w, C={C}/mo", alpha=0.85)
            data.append(ny)

    ax.set_xlabel("Annual rate (%)", fontsize=14)
    ax.set_ylabel("Years needed n", fontsize=14)
    ax.set_title("Fixed principal B & DCA C -- Annual rate vs Years n",
                 fontsize=16, fontweight='bold')
    ax.set_xlim(3, 25)
    _grid(ax, 2, 0.5, 5, 1)
    _auto_ylim(ax, data)
    _year_refs(ax, (5, 10, 15, 20, 30, 40))
    ax.legend(fontsize=7, ncol=4, loc='upper right', framealpha=0.9)
    _save(fig, "03_n_vs_annual_rate.png")


# ==================================================================
#  Chart 4: Monthly interest growth (5% annual, multiple DCA amounts)
# ==================================================================
def plot_monthly_interest_growth():
    fig, ax = plt.subplots(figsize=(18, 10))
    r_annual, r = 0.05, 0.05 / 12
    months = np.arange(1, 361)
    Ms = [1000, 2000, 3000, 5000, 8000, 10000]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']

    n_cross = np.log(2) / np.log(1 + r)
    n_cross_int = int(np.ceil(n_cross))

    max_y = max(M * ((1 + r) ** 360 - 1) for M in Ms) * 1.05
    ax.set_xlim(0, 360)
    ax.set_ylim(0, max_y)

    for M, c in zip(Ms, colors):
        interest = M * ((1 + r) ** months - 1)
        ax.plot(months, interest, color=c, lw=2.2, label=f'DCA {M:,}/mo', alpha=0.9)

        cross_val = M * ((1 + r) ** n_cross - 1)
        ax.plot(n_cross, cross_val, 'o', color=c, ms=10,
                markeredgecolor='black', markeredgewidth=1.2, zorder=5)
        ax.annotate(f'Mo {n_cross_int} = {n_cross/12:.1f}yr\nInt={cross_val:,.0f}',
                    xy=(n_cross, cross_val),
                    xytext=(n_cross + 18, cross_val + M * 0.15),
                    fontsize=8, color=c, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=c, alpha=0.85))

        ax.axhline(y=M, color=c, ls=':', lw=0.8, alpha=0.35)
        ax.text(5, M + 50, f'{M:,}', fontsize=7.5, color=c, alpha=0.7, va='bottom')

    ax.axvline(x=n_cross, color='red', ls='--', lw=1.5, alpha=0.6)
    ax.text(n_cross + 2, 50, f'Critical: Mo {n_cross_int}\n({n_cross/12:.1f}yr)',
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#fff3f3', ec='red', alpha=0.9))

    for mult in [2, 5]:
        n_m = np.log(mult + 1) / np.log(1 + r)
        ax.axvline(x=n_m, color='gray', ls=':', lw=1, alpha=0.4)
        ax.text(n_m + 2, max_y * 0.95, f'Int={mult}x DCA\n{n_m/12:.1f}yr',
                fontsize=9, color='gray', fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='gray', alpha=0.8))

    ax.set_xlabel("Month n", fontsize=14)
    ax.set_ylabel("Monthly interest", fontsize=14)
    ax.set_title(f"DCA monthly interest growth (annual return {r_annual*100:.0f}%)\n"
                 f"Marked: interest exceeds monthly DCA amount",
                 fontsize=16, fontweight='bold')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(True, which='major', ls='-', alpha=0.3)
    ax.grid(True, which='minor', ls=':', alpha=0.15)

    _top_year_axis(ax)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9,
              title='Monthly DCA', title_fontsize=12)

    summary = (f"Annual return: {r_annual*100:.0f}%  |  Monthly rate: {r*100:.4f}%\n"
               f"Interest = DCA at month {n_cross_int} ({n_cross/12:.1f} yr)\n"
               f"This critical point depends only on rate, not DCA amount")
    ax.text(0.98, 0.02, summary, transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='orange', alpha=0.9))
    _save(fig, "04_monthly_interest_growth.png")


# ==================================================================
#  Chart 5: Interest/DCA ratio across rates (normalized, with milestones)
# ==================================================================
def plot_interest_ratio():
    fig, ax = plt.subplots(figsize=(18, 10))
    annual_rates = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    cmap = plt.cm.RdYlBu_r
    colors = [cmap(i / (len(annual_rates) - 1)) for i in range(len(annual_rates))]
    months = np.arange(1, 361)
    milestones = [1, 2, 5, 10]

    for r_annual, color in zip(annual_rates, colors):
        r = r_annual / 12
        ratio = (1 + r) ** months - 1
        ax.plot(months, ratio, color=color, lw=2.5,
                label=f'{r_annual*100:.0f}% annual', alpha=0.9)
        for mult in milestones:
            n_m = np.log(mult + 1) / np.log(1 + r)
            if n_m <= 360:
                ax.plot(n_m, mult, 'o', color=color, ms=8,
                        markeredgecolor='black', markeredgewidth=0.8, zorder=5)

    for idx, (r_annual, color) in enumerate(zip(annual_rates, colors)):
        r = r_annual / 12
        n_c = np.log(2) / np.log(1 + r)
        y_off = 0.3 if idx % 2 == 0 else -0.3
        ax.annotate(f'Mo {int(np.ceil(n_c))} ({n_c/12:.1f}yr)',
                    xy=(n_c, 1.0), xytext=(n_c + 8, 1.0 + y_off),
                    fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.85))

    ref_lines = [(1, '#e74c3c', '--', 2.0, 'Interest = 1x DCA (critical)'),
                 (2, '#e67e22', '--', 1.5, '2x DCA'),
                 (5, '#8e44ad', '--', 1.5, '5x DCA'),
                 (10, '#2c3e50', '--', 1.5, '10x DCA')]
    for mult, c, ls, lw, txt in ref_lines:
        ax.axhline(y=mult, color=c, ls=ls, lw=lw, alpha=0.5)
        ax.text(5, mult * 1.15, txt, fontsize=10 if mult == 1 else 9,
                color=c, fontweight='bold', alpha=0.85)

    ax.set_xlabel('Month n', fontsize=14)
    ax.set_ylabel('Interest / Monthly DCA (ratio)', fontsize=14)
    ax.set_title('Interest-to-DCA ratio across annual return rates\n'
                 'Milestones: 1x, 2x, 5x, 10x DCA amount',
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_yscale('log')
    ax.set_ylim(0.001, 200)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:.0f}' if x >= 1 else f'{x:.2g}'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
    ax.grid(True, which='major', ls='-', alpha=0.3)
    ax.grid(True, which='minor', ls=':', alpha=0.15)

    _top_year_axis(ax)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9,
              title='Annual return', title_fontsize=12, ncol=2)

    hdr = ' ' * 6 + ''.join(f'{m}x'.rjust(9) for m in milestones)
    rows = [hdr]
    for r in annual_rates:
        cells = [f'{np.log(m+1)/np.log(1+r/12)/12:.1f}yr'.rjust(9) for m in milestones]
        rows.append(f'{r*100:>5.0f}%' + ''.join(cells))
    summary = 'Years to reach DCA multiples:\n' + '\n'.join(rows)
    ax.text(0.98, 0.02, summary, transform=ax.transAxes,
            fontsize=8, va='bottom', ha='right', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='orange', alpha=0.92))
    _save(fig, '05_interest_ratio_multi_rates.png')


# ==================================================================
#  Chart 6: Critical month vs annual return rate (lookup chart)
# ==================================================================
def plot_critical_month_vs_rate():
    fig, ax = plt.subplots(figsize=(14, 8))
    annual_rates = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    cmap = plt.cm.RdYlBu_r
    colors = [cmap(i / (len(annual_rates) - 1)) for i in range(len(annual_rates))]

    rates_dense = np.linspace(0.03, 0.25, 500)
    n_cross = np.log(2) / np.log(1 + rates_dense / 12)

    ax.plot(rates_dense * 100, n_cross, color='#2c3e50', lw=3, alpha=0.9)
    ax.plot(rates_dense * 100, n_cross / 12, color='#e74c3c', lw=3, ls='--', alpha=0.9)

    for r_annual, color in zip(annual_rates, colors):
        n_c = np.log(2) / np.log(1 + r_annual / 12)
        ax.plot(r_annual * 100, n_c, 'o', color=color, ms=12,
                markeredgecolor='black', markeredgewidth=1.2, zorder=5)
        ax.annotate(f'Mo {int(np.ceil(n_c))}\n({n_c/12:.1f}yr)',
                    xy=(r_annual * 100, n_c),
                    xytext=(r_annual * 100 + 0.8, n_c + 15),
                    fontsize=10, fontweight='bold', color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.85))

    ax.set_xlabel('Annual return rate (%)', fontsize=14)
    ax.set_ylabel('Critical month n (Interest = DCA)', fontsize=14)
    ax.set_title('Annual return vs months for interest to exceed DCA\n'
                 'Formula: n = ln(2) / ln(1 + r/12)  (independent of DCA amount)',
                 fontsize=16, fontweight='bold')
    _grid(ax, 2, 0.5, 50, 10)
    ax.set_xlim(3, 25)

    ax_right = ax.twinx()
    y_lim = ax.get_ylim()
    ax_right.set_ylim(y_lim[0] / 12, y_lim[1] / 12)
    ax_right.set_ylabel('Years', fontsize=14, color='#e74c3c')
    ax_right.tick_params(axis='y', labelcolor='#e74c3c')

    ax.legend(handles=[
        Line2D([0], [0], color='#2c3e50', lw=3, label='Months'),
        Line2D([0], [0], color='#e74c3c', lw=3, ls='--', label='Years'),
    ], fontsize=12, loc='upper right', framealpha=0.9)

    summary = ('Key takeaways:\n'
               '- Higher return = faster crossover\n'
               '- 3% needs ~23.4yr, 25% only ~3.7yr\n'
               '- Critical month depends only on rate')
    ax.text(0.98, 0.98, summary, transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='orange', alpha=0.92))
    _save(fig, '06_critical_month_vs_rate.png')


# ==================================================================
if __name__ == "__main__":
    plot_n_vs_B()
    plot_n_vs_C()
    plot_n_vs_i()
    plot_monthly_interest_growth()
    plot_interest_ratio()
    plot_critical_month_vs_rate()
