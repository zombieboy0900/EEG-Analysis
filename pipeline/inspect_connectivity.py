"""
inspect_connectivity.py — Quick connectivity diagnostics from cached data.

Reads conn_matrices.npz directly from cache (no pipeline rerun needed).
Useful for calibrating vmax_override values in plot_network_viz.

Usage
-----
    # Edge weight distributions for one subject:
    python pipeline/inspect_connectivity.py --subjects v10p

    # Network viz for one subject (uses current VMAX_NF defaults):
    python pipeline/inspect_connectivity.py --subjects v10p --network

    # Edge weight distributions for several subjects side-by-side:
    python pipeline/inspect_connectivity.py --subjects v10p v41p v43p

    # Surface Laplacian connectivity instead of no-filter:
    python pipeline/inspect_connectivity.py --subjects v10p --csd

    # Summarise 95th-percentile edge weight across all cached subjects
    # (helps you pick sensible vmax_override values):
    python pipeline/inspect_connectivity.py --summary
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

MEASURES = ['coherence', 'wpli', 'correlation']
BANDS    = list(config.FREQ_BANDS.keys())


def _load_conn(sid, csd=False):
    fname     = 'conn_csd.npz' if csd else 'conn_matrices.npz'
    cache_dir = os.path.join(config.RESULTS_ROOT, 'subjects', sid, '.cache')
    path      = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {tuple(k.split('__')): data[k] for k in data.files}


def _load_metadata():
    import csv
    gmap = {}
    with open(config.METADATA_CSV, newline='') as f:
        for row in csv.DictReader(f):
            gmap[row['subject_id'].strip()] = row['group'].strip()
    return gmap


# ══════════════════════════════════════════════════════════════════════════════
# PER-SUBJECT EDGE WEIGHT DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_distributions(sids, csd=False, out_dir=None):
    """
    For each subject in sids, generate an edge weight distribution figure.
    One figure per subject, saved to results/subjects/<sid>/diagnostics/.
    """
    gmap = _load_metadata()
    tag  = 'sl' if csd else 'nf'

    for sid in sids:
        conn = _load_conn(sid, csd=csd)
        if conn is None:
            print(f"  {sid}: no {'Surface Laplacian ' if csd else ''}conn_matrices in cache — skipping")
            continue

        group = gmap.get(sid, '?')
        n_ch  = next(iter(conn.values())).shape[0]
        n_pairs = n_ch * (n_ch - 1) // 2

        colors = {'coherence': '#1f77b4', 'wpli': '#d62728', 'correlation': '#2ca02c'}
        fig, axes = plt.subplots(3, 5, figsize=(22, 11))
        fig.subplots_adjust(hspace=0.52, wspace=0.30, left=0.07, top=0.90, bottom=0.06)

        # Per-row fixed y-axis (so bands are comparable within a measure)
        row_ymax = {}
        for m in MEASURES:
            max_count = 0
            for b in BANDS:
                key = (m, b)
                if key not in conn:
                    continue
                off_diag = conn[key][np.triu_indices(n_ch, k=1)]
                counts, _ = np.histogram(off_diag, bins=30, range=(0, 1))
                max_count = max(max_count, counts.max())
            row_ymax[m] = max_count * 1.15

        for row, m in enumerate(MEASURES):
            for col, b in enumerate(BANDS):
                ax  = axes[row, col]
                key = (m, b)
                if key not in conn:
                    ax.axis('off')
                    continue

                off_diag  = conn[key][np.triu_indices(n_ch, k=1)]
                threshold = np.percentile(off_diag, 100 * (1 - config.GRAPH_DENSITY))
                p50       = np.percentile(off_diag, 50)
                p95       = np.percentile(off_diag, 95)

                ax.hist(off_diag, bins=30, range=(0, 1),
                        color=colors[m], alpha=0.65, edgecolor='none')
                ax.axvline(threshold, color='black',   lw=1.5, ls='--', label=f'thr={threshold:.3f}')
                ax.axvline(p95,       color='#e67e22', lw=1.2, ls=':',  label=f'p95={p95:.3f}')
                ax.axvspan(threshold, 1.0, alpha=0.12, color='black')

                ax.set_xlim(0, 1)
                ax.set_ylim(0, row_ymax.get(m, 1))
                ax.tick_params(labelsize=6)
                ax.set_xlabel('edge weight', fontsize=6)
                ax.set_ylabel('count',       fontsize=6)

                ax.text(0.97, 0.97,
                        f'p50={p50:.3f}\np95={p95:.3f}',
                        transform=ax.transAxes, fontsize=6,
                        ha='right', va='top', color='#333333')

                if row == 0:
                    ax.set_title(b, fontsize=10, fontweight='bold')
                if col == 0:
                    ax.text(-0.38, 0.5, m, transform=ax.transAxes,
                            fontsize=10, fontweight='bold', va='center', rotation=90)

        fig.suptitle(
            f'Edge Weight Distributions — {sid} ({group})  |  '
            f'{"Surface Laplacian" if csd else "No Filter"}\n'
            f'Dashed = graph threshold (top {int(config.GRAPH_DENSITY*100)}%)  |  '
            f'Orange dotted = 95th percentile  |  '
            f'{n_pairs} pairwise edges per matrix',
            fontsize=11,
        )

        save_dir = out_dir or os.path.join(
            config.RESULTS_ROOT, 'subjects', sid, 'diagnostics')
        os.makedirs(save_dir, exist_ok=True)
        fname = f'edge_weight_dist_{tag}.png'
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {sid}: saved {fname}  →  {save_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP EDGE WEIGHT DISTRIBUTIONS (ADHD vs TDC)
# ══════════════════════════════════════════════════════════════════════════════

def _build_pool(csd=False):
    """Load and pool all edge weights per group/measure/band. Returns (pool, counts)."""
    gmap   = _load_metadata()
    pool   = {
        'ADHD': {m: {b: [] for b in BANDS} for m in MEASURES},
        'TDC':  {m: {b: [] for b in BANDS} for m in MEASURES},
    }
    counts = {'ADHD': 0, 'TDC': 0}
    for sid, grp in sorted(gmap.items()):
        if grp not in pool:
            continue
        conn = _load_conn(sid, csd=csd)
        if conn is None:
            continue
        counts[grp] += 1
        n_ch = next(iter(conn.values())).shape[0]
        for m in MEASURES:
            for b in BANDS:
                key = (m, b)
                if key not in conn:
                    continue
                pool[grp][m][b].append(conn[key][np.triu_indices(n_ch, k=1)])
    return pool, counts


def _compute_limits(pool, n_bins=40):
    """
    Compute per-row (per-measure) x and y limits from a pool.
    Returns (row_xmax, row_ymax) dicts keyed by measure name.
    """
    row_xmax = {}
    row_ymax = {}
    for m in MEASURES:
        all_vals = [v for b in BANDS for grp in ['ADHD', 'TDC']
                    for v in (pool[grp][m][b] if pool[grp][m][b] else [])]
        if not all_vals:
            row_xmax[m] = 1.0
            row_ymax[m] = 1.0
            continue
        arr  = np.concatenate(all_vals)
        xmax = max(float(np.percentile(arr, 99.5)), 0.05)
        ymax = 0.0
        for b in BANDS:
            for grp in ['ADHD', 'TDC']:
                if not pool[grp][m][b]:
                    continue
                c, _ = np.histogram(np.concatenate(pool[grp][m][b]),
                                    bins=n_bins, range=(0, xmax), density=True)
                ymax = max(ymax, float(c.max()))
        row_xmax[m] = xmax
        row_ymax[m] = ymax * 1.18
    return row_xmax, row_ymax


def _draw_group_figure(pool, counts, row_xmax, row_ymax, csd, out_dir, n_bins=40):
    """Draw and save the 3×5 group distribution figure using the given limits."""
    tag   = 'sl' if csd else 'nf'
    label = 'Surface Laplacian' if csd else 'No Filter'

    ADHD_COLOR = '#e74c3c'
    TDC_COLOR  = '#3498db'
    MEASURE_LABELS = {'coherence': 'Coherence', 'wpli': 'wPLI', 'correlation': 'Correlation'}
    BAND_LABELS    = {'delta': 'δ Delta', 'theta': 'θ Theta', 'alpha': 'α Alpha',
                      'beta': 'β Beta',  'gamma': 'γ Gamma'}

    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    fig.subplots_adjust(hspace=0.45, wspace=0.30, left=0.06, top=0.91, bottom=0.07)

    for row, m in enumerate(MEASURES):
        xmax = row_xmax[m]
        ymax = row_ymax[m]

        for col, b in enumerate(BANDS):
            ax = axes[row, col]

            for grp, color in [('ADHD', ADHD_COLOR), ('TDC', TDC_COLOR)]:
                if not pool[grp][m][b]:
                    continue
                pooled   = np.concatenate(pool[grp][m][b])
                n_subj   = len(pool[grp][m][b])
                mean_val = float(np.mean(pooled))
                p95_val  = float(np.percentile(pooled, 95))

                ax.hist(pooled, bins=n_bins, range=(0, xmax), density=True,
                        color=color, alpha=0.45, edgecolor='none',
                        label=f'{grp} (n={n_subj})')
                ax.axvline(mean_val, color=color, lw=1.5, ls='-',  alpha=0.9)
                ax.axvline(p95_val,  color=color, lw=1.0, ls='--', alpha=0.7)
                ax.text(p95_val, ymax * (0.88 if grp == 'ADHD' else 0.72),
                        f'p95={p95_val:.3f}', color=color, fontsize=5.5,
                        ha='center', va='top', rotation=90)

            all_pooled = np.concatenate(
                pool['ADHD'][m][b] + pool['TDC'][m][b]
            ) if (pool['ADHD'][m][b] or pool['TDC'][m][b]) else np.array([0])
            thr = float(np.percentile(all_pooled, 100 * (1 - config.GRAPH_DENSITY)))
            ax.axvline(thr, color='#444444', lw=1.1, ls=':', alpha=0.65)
            ax.text(thr, ymax * 0.99, f' thr\n {thr:.3f}',
                    color='#444444', fontsize=5, va='top')

            ax.set_xlim(0, xmax)
            ax.set_ylim(0, ymax)
            ax.tick_params(labelsize=6)
            ax.set_xlabel('edge weight', fontsize=7)
            ax.grid(True, axis='y', linewidth=0.3, alpha=0.35)

            if row == 0:
                ax.set_title(BAND_LABELS[b], fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel('density', fontsize=7)
                ax.text(-0.38, 0.5, MEASURE_LABELS[m], transform=ax.transAxes,
                        fontsize=10, fontweight='bold', va='center', rotation=90)
            if col == 0:
                ax.legend(fontsize=7, loc='upper right', framealpha=0.85)

    fig.suptitle(
        f'Edge Weight Distributions — ADHD (n={counts["ADHD"]}) vs TDC (n={counts["TDC"]})'
        f'  |  {label}\n'
        f'Pooled across subjects, normalised to density  |  '
        f'Solid = group mean  |  Dashed = group p95  |  Dotted = graph threshold  |  '
        f'Axes matched to Raw figure for direct comparison',
        fontsize=10,
    )

    os.makedirs(out_dir, exist_ok=True)
    fname = f'edge_dist_group_{tag}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}  →  {out_dir}/")


def _draw_combined_figure(pool_nf, pool_sl, counts_nf, counts_sl,
                          shared_xmax, shared_ymax, out_dir, n_bins=40):
    """
    Single 6×5 figure with no-filter and surface laplacian rows interleaved per measure:
      Row 0 — Coherence  No Filter
      Row 1 — Coherence  Surface Laplacian
      Row 2 — wPLI       No Filter
      Row 3 — wPLI       Surface Laplacian
      Row 4 — Correlation No Filter
      Row 5 — Correlation Surface Laplacian
    Axes are shared across all 30 panels within each measure pair, so the
    effect of the spatial filter is immediately visible by comparing adjacent rows.
    """
    ADHD_COLOR = '#e74c3c'
    TDC_COLOR  = '#3498db'
    BAND_LABELS = {'delta': 'δ Delta', 'theta': 'θ Theta', 'alpha': 'α Alpha',
                   'beta': 'β Beta',   'gamma': 'γ Gamma'}

    # Row layout: (measure, pool, counts, label, background_color)
    ROW_DEFS = []
    for m in MEASURES:
        ROW_DEFS.append((m, pool_nf, counts_nf, 'No Filter',         '#f7f7f7'))
        ROW_DEFS.append((m, pool_sl, counts_sl, 'Surface Laplacian', '#eaf4fb'))

    n_rows = len(ROW_DEFS)   # 6
    fig, axes = plt.subplots(n_rows, 5, figsize=(22, 18))
    fig.subplots_adjust(hspace=0.18, wspace=0.28, left=0.08, top=0.94, bottom=0.05)

    MEASURE_LABELS = {'coherence': 'Coherence', 'wpli': 'wPLI', 'correlation': 'Correlation'}

    for row_idx, (m, pool, counts, pipeline_tag, bg) in enumerate(ROW_DEFS):
        xmax = shared_xmax[m]
        ymax = shared_ymax[m]

        for col, b in enumerate(BANDS):
            ax = axes[row_idx, col]
            ax.set_facecolor(bg)

            for grp, color in [('ADHD', ADHD_COLOR), ('TDC', TDC_COLOR)]:
                if not pool[grp][m][b]:
                    continue
                pooled   = np.concatenate(pool[grp][m][b])
                n_subj   = len(pool[grp][m][b])
                mean_val = float(np.mean(pooled))
                p95_val  = float(np.percentile(pooled, 95))

                ax.hist(pooled, bins=n_bins, range=(0, xmax), density=True,
                        color=color, alpha=0.45, edgecolor='none',
                        label=f'{grp} (n={n_subj})')
                ax.axvline(mean_val, color=color, lw=1.4, ls='-',  alpha=0.9)
                ax.axvline(p95_val,  color=color, lw=0.9, ls='--', alpha=0.7)

            all_pooled = np.concatenate(
                pool['ADHD'][m][b] + pool['TDC'][m][b]
            ) if (pool['ADHD'][m][b] or pool['TDC'][m][b]) else np.array([0])
            thr = float(np.percentile(all_pooled, 100 * (1 - config.GRAPH_DENSITY)))
            ax.axvline(thr, color='#444444', lw=1.0, ls=':', alpha=0.6)

            ax.set_xlim(0, xmax)
            ax.set_ylim(0, ymax)
            ax.tick_params(labelsize=5)
            ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)

            # Band labels only on top row
            if row_idx == 0:
                ax.set_title(BAND_LABELS[b], fontsize=9, fontweight='bold')

            # x-axis label only on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel('edge weight', fontsize=6)

            # Row label on leftmost panel
            if col == 0:
                measure_str = MEASURE_LABELS[m]
                ax.set_ylabel('density', fontsize=6)
                ax.text(-0.42, 0.5,
                        f'{measure_str}\n{pipeline_tag}',
                        transform=ax.transAxes,
                        fontsize=8, fontweight='bold', va='center',
                        rotation=90,
                        color='#1a6f00' if pipeline_tag == 'Surface Laplacian' else '#333333')

            # Thin divider line between measure pairs (between SL row and next NF row)
            if pipeline_tag == 'Surface Laplacian' and row_idx < n_rows - 1:
                ax.spines['bottom'].set_linewidth(1.8)
                ax.spines['bottom'].set_color('#aaaaaa')

            # Legend only on top-left panel
            if row_idx == 0 and col == 0:
                ax.legend(fontsize=6, loc='upper right', framealpha=0.85)

    fig.suptitle(
        f'Edge Weight Distributions — No Filter vs Surface Laplacian  |  '
        f'ADHD (n={counts_nf["ADHD"]}) vs TDC (n={counts_nf["TDC"]})\n'
        f'Each measure pair shares axes (no filter=grey, surface laplacian=blue tint)  |  '
        f'Solid = group mean  |  Dashed = p95  |  Dotted = graph threshold',
        fontsize=10,
    )

    os.makedirs(out_dir, exist_ok=True)
    fname = 'edge_dist_group_combined.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}  →  {out_dir}/")


def plot_group_distributions():
    """
    Build both no-filter and surface laplacian pools, compute shared axes, then save:
      - edge_dist_group_nf.png       (15 panels, no filter only)
      - edge_dist_group_sl.png       (15 panels, surface laplacian only)
      - edge_dist_group_combined.png (30 panels, no filter + surface laplacian interleaved)
    All three figures use identical axes so any two can be compared directly.
    """
    print("Loading no-filter connectivity...")
    pool_nf, counts_nf = _build_pool(csd=False)
    print("Loading surface laplacian connectivity...")
    pool_sl, counts_sl = _build_pool(csd=True)

    if counts_nf['ADHD'] == 0 and counts_nf['TDC'] == 0:
        print("No cached no-filter connectivity found.")
        return

    lims_nf = _compute_limits(pool_nf)
    lims_sl = _compute_limits(pool_sl)

    # Shared limits: max across both pipelines per measure
    shared_xmax = {m: max(lims_nf[0][m], lims_sl[0].get(m, 0)) for m in MEASURES}
    shared_ymax = {m: max(lims_nf[1][m], lims_sl[1].get(m, 0)) for m in MEASURES}

    figs          = os.path.join(config.RESULTS_ROOT, 'group', 'figures')
    no_filter_ed  = os.path.join(figs, 'no_filter', 'edge_dist')
    sl_ed         = os.path.join(figs, 'surface_laplacian', 'edge_dist')
    comb_ed       = os.path.join(figs, 'overview', 'edge_dist', 'combined')
    for d in [no_filter_ed, sl_ed, comb_ed]:
        os.makedirs(d, exist_ok=True)

    print("Drawing no-filter figure...")
    _draw_group_figure(pool_nf, counts_nf, shared_xmax, shared_ymax,
                       csd=False, out_dir=no_filter_ed)

    has_sl = counts_sl['ADHD'] > 0 or counts_sl['TDC'] > 0
    if has_sl:
        print("Drawing surface laplacian figure...")
        _draw_group_figure(pool_sl, counts_sl, shared_xmax, shared_ymax,
                           csd=True, out_dir=sl_ed)
        print("Drawing combined figure...")
        _draw_combined_figure(pool_nf, pool_sl, counts_nf, counts_sl,
                              shared_xmax, shared_ymax, comb_ed)
    else:
        print("No surface laplacian connectivity cached — skipping surface laplacian and combined figures.")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET-WIDE SUMMARY (helps pick vmax_override)
# ══════════════════════════════════════════════════════════════════════════════

def summarise_vmax(csd=False):
    """
    Print 50th/95th/99th percentile of edge weights across ALL cached subjects.
    Use the 95th percentile as your vmax_override — nearly all edges will be
    visible without the colormap being wasted on rare outliers.
    """
    gmap  = _load_metadata()
    stats = {m: {b: [] for b in BANDS} for m in MEASURES}
    found = 0

    for sid in sorted(gmap):
        conn = _load_conn(sid, csd=csd)
        if conn is None:
            continue
        found += 1
        n_ch = next(iter(conn.values())).shape[0]
        for m in MEASURES:
            for b in BANDS:
                key = (m, b)
                if key not in conn:
                    continue
                off_diag = conn[key][np.triu_indices(n_ch, k=1)]
                stats[m][b].append(off_diag)

    if found == 0:
        print("No cached connectivity data found.")
        return

    print(f"\n{'='*72}")
    print(f"CONNECTIVITY EDGE WEIGHT SUMMARY  —  "
          f"{'Surface Laplacian' if csd else 'No Filter'}  —  {found} subjects")
    print(f"{'='*72}")
    print(f"\n  Use the p95 column as vmax_override to calibrate network_viz.\n")
    print(f"  {'Measure':<12} {'Band':<8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    print(f"  {'-'*12} {'-'*8} {'------':>8} {'------':>8} {'------':>8} {'------':>8}")

    suggested = {}
    for m in MEASURES:
        max_p95 = 0.0
        for b in BANDS:
            all_vals = np.concatenate(stats[m][b]) if stats[m][b] else np.array([0])
            p50  = np.percentile(all_vals, 50)
            p95  = np.percentile(all_vals, 95)
            p99  = np.percentile(all_vals, 99)
            vmax = all_vals.max()
            print(f"  {m:<12} {b:<8} {p50:>8.4f} {p95:>8.4f} {p99:>8.4f} {vmax:>8.4f}")
            max_p95 = max(max_p95, p95)
        suggested[m] = round(max_p95, 2)
        print()

    print(f"  Suggested vmax_override (p95 across all bands):")
    tag = 'VMAX_SL' if csd else 'VMAX_NF'
    items = ', '.join(f"'{m}': {v}" for m, v in suggested.items())
    print(f"    {tag} = {{{items}}}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connectivity diagnostics from cache.')
    parser.add_argument('--subjects', nargs='+', metavar='SID',
                        help='Subject IDs to inspect (e.g. v10p v41p)')
    parser.add_argument('--csd',     action='store_true',
                        help='Use Surface Laplacian connectivity instead of no-filter')
    parser.add_argument('--summary', action='store_true',
                        help='Print dataset-wide edge weight percentiles and suggested vmax values')
    parser.add_argument('--group',   action='store_true',
                        help='Plot ADHD vs TDC edge weight distributions (no filter + surface laplacian, shared axes)')
    parser.add_argument('--out-dir', metavar='DIR',
                        help='Override output directory for figures')
    args = parser.parse_args()

    if args.summary:
        summarise_vmax(csd=args.csd)

    if args.group:
        # Always produces both no-filter and surface laplacian figures with matched axes
        plot_group_distributions()

    if args.subjects:
        plot_distributions(args.subjects, csd=args.csd, out_dir=args.out_dir)

    if not args.summary and not args.group and not args.subjects:
        plot_group_distributions()
