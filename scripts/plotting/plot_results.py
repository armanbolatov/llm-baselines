"""Shared pieces for the three figures of the paper.

  plot_124m.py              loss against FLOPs at 124M
  plot_loss_vs_cost.py      355M against FLOPs, and 124M against bytes per step
  plot_norm_diagnostics.py  the constants of the bound during training

Everything here is either the visual style, so the three look like one set, or the FLOPs
accounting, so the 124M and 355M axes are computed the same way.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# --- Visual style ---
# SignMuon family: red gradient, darker = lower K (more NS), triangle marker
# LionMuon family: blue gradient, darker = lower K, circle marker
# AdamW: gray, square marker
# Darker color = more frequent NS steps (lower K)

STYLE = {
    'AdamW':              {'color': '#888888', 'ls': '-',  'lw': 1.8, 'marker': 's'},
    # SignMuon family: one red hue, dark (P=1) to light (P=inf), in even lightness steps
    'Muon (P=1)':         {'color': '#5a0a0a', 'ls': '-',        'lw': 2.2, 'marker': '^'},
    'SignMuon P=2':       {'color': '#c8202a', 'ls': '--',       'lw': 2.2, 'marker': '^'},
    'SignMuon P=5':       {'color': '#ee6a6a', 'ls': '-.',       'lw': 2.2, 'marker': '^'},
    'SignMuon P=20':      {'color': '#f5a3a3', 'ls': ':',        'lw': 2.2, 'marker': '^'},
    'SignMuon P=100':     {'color': '#f8bcbc', 'ls': (0,(1,3)),  'lw': 2.2, 'marker': '^'},
    'Signum (P=\u221e)':  {'color': '#fad4d4', 'ls': (0,(5,10)), 'lw': 2.2, 'marker': '^'},
    # LionMuon family: one blue hue, dark (P=1) to light (P=inf), in even lightness steps
    'LionMuon P=1':        {'color': '#0b1f6b', 'ls': '-',        'lw': 2.2, 'marker': 'o'},
    'LionMuon P=2':        {'color': '#1f5fd6', 'ls': '--',       'lw': 2.2, 'marker': 'o'},
    'LionMuon P=5':        {'color': '#4f9be8', 'ls': '-.',       'lw': 2.2, 'marker': 'o'},
    'LionMuon P=20':       {'color': '#8fc1f5', 'ls': ':',        'lw': 2.2, 'marker': 'o'},
    'LionMuon P=100':      {'color': '#aed3f8', 'ls': (0,(1,3)),  'lw': 2.2, 'marker': 'o'},
    'Lion (P=\u221e)':   {'color': '#cce2fa', 'ls': (0,(5,10)), 'lw': 2.2, 'marker': 'o'},
}


def compute_flops(n_layer, n_embd, seq_len, batch_size, iterations, opt, K=1, ns_steps=6,
                  vocab_size=50304, opt_diag=None):
    params_per_layer = 4 * n_embd**2 + 8 * n_embd**2
    embed_params = vocab_size * n_embd + seq_len * n_embd
    n_params = n_layer * params_per_layer + embed_params
    model_flops_per_iter = 6 * n_params * seq_len * batch_size

    # If exact per-run optimizer diagnostics are present, use them.
    if isinstance(opt_diag, dict) and opt_diag:
        ns_total = float(opt_diag.get('ns_flops', 0.0) or 0.0)
        sign_total = float(opt_diag.get('sign_flops', 0.0) or 0.0)
        if (ns_total + sign_total) > 0.0:
            total = iterations * model_flops_per_iter + ns_total + sign_total
            return total, total

    d = n_embd
    ns_flops_per_layer = 0
    matvec_mn_sum_per_layer = 0
    for (m, n) in [(d, 3*d), (d, d), (d, 4*d), (4*d, d)]:
        dmin, dmax = min(m, n), max(m, n)
        ns_flops_per_layer += ns_steps * 2 * dmin**2 * (2*dmax + dmin)
        matvec_mn_sum_per_layer += m * n
    ns_flops_per_iter = n_layer * ns_flops_per_layer
    sign_flops_per_iter = n_layer * matvec_mn_sum_per_layer

    if opt in ('adamw', 'lion'):
        total = iterations * model_flops_per_iter
        return total, total
    elif opt in ('muon', 'sign_muon', 'lion_muon'):
        if K <= 1:
            total = iterations * (model_flops_per_iter + ns_flops_per_iter)
            return total, total
        p_muon = 1.0 / K
        extra = p_muon * ns_flops_per_iter + (1.0 - p_muon) * sign_flops_per_iter
        total = iterations * (model_flops_per_iter + extra)
        return total, total
    total = iterations * model_flops_per_iter
    return total, total



# ---------------------------------------------------------------------------
# One look for the scatter figures (plot_124m.py, plot_loss_vs_cost.py), so a
# reader moving between them sees the same markers, sizes, grid and legend.
# ---------------------------------------------------------------------------

STYLE_EXTRA = {                      # baselines that are not part of the two families
    'Dion':         {'color': '#2e8b57', 'marker': 'D'},
    'Dion, rank 1/2': {'color': '#8fd19e', 'marker': 'D'},   # same family, lighter: it sends more
    'MuonBP (P=5)': {'color': '#e69500', 'marker': 'v'},
}
POINT_KW = dict(s=110, zorder=5, alpha=0.9, edgecolor='black', linewidth=0.5)
ERR_KW = dict(zorder=4, capsize=3, linewidth=1)
LEGEND_KW = dict(loc='lower center', fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
LABEL_FS, TITLE_FS = 11, 12


def style_for(label):
    """Marker and colour for a legend label, from either table."""
    s = STYLE.get(label)
    if s is None:
        key = next((k for k in STYLE_EXTRA if label.startswith(k)), None)
        s = STYLE_EXTRA[key] if key else None
    if s is None:
        raise KeyError(f"no style for {label!r}")
    return s


def finish_axis(ax, xlabel, title=None):
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    if title:
        ax.set_title(title, fontsize=TITLE_FS)
    ax.margins(x=0.06, y=0.15)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


def legend_handle(label):
    s = style_for(label)
    return Line2D([0], [0], marker=s['marker'], linestyle='None', markerfacecolor=s['color'],
                  markeredgecolor='black', markeredgewidth=0.5, markersize=8, label=label)
