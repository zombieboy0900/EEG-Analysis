"""
pipeline_group.py — group-level analysis for the ADHD EEG TDA pipeline.

Loads per-subject betti_features.pkl files, merges with metadata.csv,
and produces group figures + logistic regression results.

Can be run at any time — subjects that haven't finished yet are skipped.

Usage:
    python pipeline/pipeline_group.py
    python pipeline/pipeline_group.py --force   # regenerate even if outputs exist
"""

import os
import sys
import re
import csv
import pickle
import argparse
import warnings
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

warnings.filterwarnings('ignore')

MEASURES = ['coherence', 'wpli', 'correlation']
BANDS    = ['delta', 'theta', 'alpha', 'beta', 'gamma']
MEASURE_LABELS = {'coherence': 'Coherence', 'wpli': 'wPLI', 'correlation': 'Correlation'}
BAND_LABELS    = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α', 'beta': 'β', 'gamma': 'γ'}

ADHD_COLOR = '#e74c3c'   # red
TDC_COLOR  = '#3498db'   # blue


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_subjects():
    """
    Load metadata.csv and betti_features.pkl for every completed subject.

    Returns
    -------
    subjects : list of dicts, one per loaded subject
        Keys: subject_id, group, age, sex, features
              where features is the betti_features dict from pkl.
    skipped  : list of subject_ids that had no betti_features.pkl yet.
    """
    if not os.path.exists(config.METADATA_CSV):
        raise FileNotFoundError(f"metadata.csv not found at {config.METADATA_CSV}")

    subjects = []
    skipped  = []

    with open(config.METADATA_CSV, newline='') as f:
        for row in csv.DictReader(f):
            sid      = row['subject_id'].strip()
            pkl_path = os.path.join(config.RESULTS_ROOT, 'subjects', sid, 'betti_features.pkl')

            if not os.path.exists(pkl_path):
                skipped.append(sid)
                continue

            try:
                with open(pkl_path, 'rb') as pf:
                    features = pickle.load(pf)
            except Exception as e:
                print(f"  WARNING: Could not load {pkl_path}: {e}")
                skipped.append(sid)
                continue

            # Also load Surface Laplacian features if available
            sl_pkl = os.path.join(config.RESULTS_ROOT, 'subjects', sid, 'betti_features_csd.pkl')
            features_sl = None
            if os.path.exists(sl_pkl):
                try:
                    with open(sl_pkl, 'rb') as pf:
                        features_sl = pickle.load(pf)
                except Exception:
                    pass

            subjects.append({
                'subject_id':  sid,
                'group':       row.get('group', '').strip(),
                'features':    features,
                'features_sl': features_sl,
            })

    return subjects, skipped


def _feat(subject, pipeline):
    """Return the right features dict for 'no_filter' or 'surface_laplacian' pipeline."""
    return subject['features_sl'] if pipeline == 'surface_laplacian' else subject['features']


def _get_curve(subjects, group, measure, band, homer, pipeline='no_filter'):
    """Return (n_subjects, 100) array of Betti curves for one group/measure/band."""
    key   = (measure, band)
    field = f'betti_{homer}'
    curves = []
    for s in subjects:
        feat = _feat(s, pipeline)
        if feat is None:
            continue
        if s['group'] == group and key in feat:
            c = np.array(feat[key].get(field, []))
            if c.shape == (config.N_FILT_STEPS,):
                curves.append(c)
    return np.array(curves) if curves else np.zeros((0, config.N_FILT_STEPS))


def _get_scalar(subjects, group, measure, band, field, pipeline='no_filter'):
    """Return list of scalar feature values for one group/measure/band."""
    key = (measure, band)
    return [
        _feat(s, pipeline)[key][field]
        for s in subjects
        if _feat(s, pipeline) is not None
        and s['group'] == group
        and key in _feat(s, pipeline)
        and field in _feat(s, pipeline)[key]
    ]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Group Betti curves ± 95% CI
# ══════════════════════════════════════════════════════════════════════════════

def plot_group_betti_curves(subjects, out_dir, pipeline='no_filter'):
    """
    3×5 grid of mean Betti curves ± 95% CI, ADHD vs TDC.
    One figure for B0, one for B1.
    pipeline: 'no_filter' (no spatial filter) or 'surface_laplacian' (surface Laplacian).
    out_dir should already be the pipeline-specific subdirectory.
    """
    pipeline_label = 'No Filter' if pipeline == 'no_filter' else 'Surface Laplacian'

    eps      = np.linspace(0, 1.0, config.N_FILT_STEPS)
    n_adhd   = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc    = sum(1 for s in subjects if s['group'] == 'TDC')

    for homer in ['0', '1']:
        fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharey=False)
        fig.subplots_adjust(hspace=0.38, wspace=0.28, top=0.88, bottom=0.06,
                            left=0.06, right=0.98)

        # Pre-compute global y-max across all panels for this homer
        global_ymax = 0
        all_means   = []
        for m in MEASURES:
            for b in BANDS:
                for grp in ['ADHD', 'TDC']:
                    c = _get_curve(subjects, grp, m, b, homer, pipeline)
                    if c.shape[0] > 0:
                        mean = c.mean(axis=0)
                        global_ymax = max(global_ymax, float(mean.max()))
                        all_means.append(mean)
        if global_ymax == 0:
            global_ymax = 1.0

        # For B1: zoom x-axis to where signal actually lives
        # Find indices where any mean curve exceeds 3% of global_ymax
        if homer == '1' and all_means:
            threshold  = 0.03 * global_ymax
            active     = np.zeros(config.N_FILT_STEPS, dtype=bool)
            for mean in all_means:
                active |= (mean > threshold)
            active_idx = np.where(active)[0]
            if len(active_idx) > 0:
                pad_steps = max(3, int(config.N_FILT_STEPS * 0.05))
                x_lo = eps[max(0, active_idx[0]  - pad_steps)]
                x_hi = eps[min(config.N_FILT_STEPS - 1, active_idx[-1] + pad_steps)]
            else:
                x_lo, x_hi = 0.0, 1.0
        else:
            x_lo, x_hi = 0.0, 1.0

        for row, m in enumerate(MEASURES):
            for col, b in enumerate(BANDS):
                ax = axes[row, col]

                for grp, color in [('ADHD', ADHD_COLOR), ('TDC', TDC_COLOR)]:
                    curves = _get_curve(subjects, grp, m, b, homer, pipeline)
                    n = curves.shape[0]
                    if n == 0:
                        continue

                    mean = curves.mean(axis=0)
                    if n > 1:
                        se   = curves.std(axis=0, ddof=1) / np.sqrt(n)
                        ci   = 1.96 * se
                    else:
                        ci = np.zeros_like(mean)

                    ax.plot(eps, mean, color=color, linewidth=1.8,
                            label=f'{grp} (n={n})')
                    ax.fill_between(eps, mean - ci, mean + ci,
                                    color=color, alpha=0.18)

                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(0, global_ymax * 1.08)
                ax.set_xlabel('ε', fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, linewidth=0.3, alpha=0.5)

                if row == 0:
                    ax.set_title(BAND_LABELS[b], fontsize=11, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(MEASURE_LABELS[m], fontsize=8)

        # Single legend in top-left panel
        axes[0, 0].legend(fontsize=7, loc='upper right', framealpha=0.8)

        fig.suptitle(
            f'Group Average Betti-{homer} Curves ± 95% CI  —  {pipeline_label}\n'
            f'ADHD (n={n_adhd}) vs TDC (n={n_tdc})  —  '
            f'Descriptive only; see auc_regression.txt for statistics',
            fontsize=12,
        )

        fname = f'betti_b{homer}_adhd_vs_tdc.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — AUC boxplots
# ══════════════════════════════════════════════════════════════════════════════

def plot_auc_distributions(subjects, out_dir, pipeline='no_filter'):
    """
    One figure per Betti dimension (B0, B1).
    Layout: 3 rows (measures) × 5 columns (bands) — mirrors Betti curve figures.
    Small n (< 15/group): dots + mean line.
    Large n (≥ 15/group): boxplot + jitter.
    pipeline: 'no_filter' or 'surface_laplacian'.
    out_dir should already be the pipeline-specific subdirectory.
    """
    pipeline_label = 'No Filter' if pipeline == 'no_filter' else 'Surface Laplacian'

    n_adhd      = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc       = sum(1 for s in subjects if s['group'] == 'TDC')
    use_boxplot = min(n_adhd, n_tdc) >= 15
    rng         = np.random.default_rng(42)
    mode_note   = 'Boxplot + jitter' if use_boxplot else 'Dots = subjects  |  Line = mean'

    for homer in ['0', '1']:
        field = f'auc_b{homer}'

        # Per-row (per-measure) y-range — prevents wPLI outliers from
        # compressing the coherence/correlation panels
        row_ylims = {}
        for m in MEASURES:
            row_vals = []
            for b in BANDS:
                for grp in ['ADHD', 'TDC']:
                    row_vals.extend(_get_scalar(subjects, grp, m, b, field, pipeline))
            if row_vals:
                rmin = min(row_vals)
                rmax = max(row_vals)
                pad  = (rmax - rmin) * 0.18 if rmax > rmin else 0.5
                row_ylims[m] = (rmin - pad, rmax + pad)
            else:
                row_ylims[m] = (0, 1)

        fig, axes = plt.subplots(3, 5, figsize=(16, 10))
        fig.subplots_adjust(hspace=0.40, wspace=0.35, top=0.88, bottom=0.07,
                            left=0.07, right=0.98)

        for row, m in enumerate(MEASURES):
            for col, b in enumerate(BANDS):
                ax        = axes[row, col]
                adhd_vals = _get_scalar(subjects, 'ADHD', m, b, field, pipeline)
                tdc_vals  = _get_scalar(subjects, 'TDC',  m, b, field, pipeline)

                if use_boxplot:
                    data_to_plot = [x for x in [adhd_vals, tdc_vals] if x]
                    colors_box   = ([ADHD_COLOR] if adhd_vals else []) + ([TDC_COLOR] if tdc_vals else [])
                    labels       = (['ADHD'] if adhd_vals else []) + (['TDC'] if tdc_vals else [])
                    if data_to_plot:
                        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                                        widths=0.5, showfliers=False)
                        for patch, color in zip(bp['boxes'], colors_box):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.5)
                        for element in ['whiskers', 'caps', 'medians']:
                            for line in bp[element]:
                                line.set_color('#333333')
                                line.set_linewidth(1.2)
                        for vals, x_pos, color in zip(data_to_plot, range(1, 3), colors_box):
                            jitter = rng.uniform(-0.15, 0.15, len(vals))
                            ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                                       color=color, s=18, alpha=0.7, zorder=3)
                else:
                    for vals, x_center, color in [
                            (adhd_vals, 1, ADHD_COLOR),
                            (tdc_vals,  2, TDC_COLOR)]:
                        if not vals:
                            continue
                        jitter = rng.uniform(-0.2, 0.2, len(vals))
                        ax.scatter(np.full(len(vals), x_center) + jitter, vals,
                                   color=color, s=40, alpha=0.85, zorder=3,
                                   edgecolors='white', linewidths=0.5)
                        ax.plot([x_center - 0.32, x_center + 0.32],
                                [np.mean(vals), np.mean(vals)],
                                color=color, linewidth=2.2, zorder=4)
                    ax.set_xticks([1, 2])
                    ax.set_xticklabels(['ADHD', 'TDC'], fontsize=8)

                ax.set_xlim(0.3, 2.7)
                ax.set_ylim(*row_ylims[m])
                ax.tick_params(labelsize=8)
                ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

                if row == 0:
                    ax.set_title(BAND_LABELS[b], fontsize=12, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f'{MEASURE_LABELS[m]}\nAUC', fontsize=9)

        fig.suptitle(
            f'B{homer} AUC Distributions — ADHD (n={n_adhd}) vs TDC (n={n_tdc})\n'
            f'{pipeline_label}  |  {mode_note}',
            fontsize=12,
        )

        fname = f'auc_b{homer}_distributions.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Age distribution diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def plot_group_counts(subjects, out_dir):
    """
    Histogram of age by group. Confirms ADHD skews younger — motivates
    including age as covariate in all regression models.
    """
    n_adhd = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc  = sum(1 for s in subjects if s['group'] == 'TDC')

    fig, ax = plt.subplots(figsize=(5, 4))
    groups  = ['ADHD', 'TDC']
    counts  = [n_adhd, n_tdc]
    colors  = [ADHD_COLOR, TDC_COLOR]
    bars    = ax.bar(groups, counts, color=colors, alpha=0.75, edgecolor='white', width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of subjects', fontsize=11)
    ax.set_title(
        'Subjects per Group\n'
        'Note: individual age/sex not available in this dataset',
        fontsize=11,
    )
    ax.set_ylim(0, max(counts) * 1.2 if counts else 1)
    ax.grid(True, axis='y', linewidth=0.4, alpha=0.5)

    fname = 'group_counts.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS — Logistic regression matching Gracia-Tabuenca 2020 Table 2
# ══════════════════════════════════════════════════════════════════════════════

def run_group_statistics(subjects, out_dir, pipeline='no_filter'):
    """
    Logistic regression per (measure, band, homer):
        group ~ auc + slope + kurtosis

    Matches Gracia-Tabuenca 2020 methodology.
    Results written to out_dir/auc_regression.txt.
    pipeline: 'no_filter' or 'surface_laplacian'.
    out_dir should already be the pipeline-specific subdirectory.

    Requires statsmodels. sklearn used for cross-validated ROC AUC.
    """
    pipeline_label = 'No Filter' if pipeline == 'no_filter' else 'Surface Laplacian'

    try:
        import statsmodels.api as sm
    except ImportError:
        print("  Skipping statistics — statsmodels not installed. Run: pip install statsmodels")
        return

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        sklearn_available = True
    except ImportError:
        sklearn_available = False

    n_total = len(subjects)
    n_adhd  = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc   = sum(1 for s in subjects if s['group'] == 'TDC')

    lines = [
        'ADHD EEG TDA Pipeline — Group Logistic Regression',
        f'Pipeline: {pipeline_label}',
        'Method: Logistic regression based on Gracia-Tabuenca 2020 (eNeuro)',
        f'Model:  group ~ auc + slope + kurtosis',
        f'N:      {n_total} subjects  (ADHD={n_adhd}, TDC={n_tdc})',
        '',
        'NOTE: Individual age/sex not available in this dataset.',
        '      Gracia-Tabuenca 2020 included age/sex as covariates — results',
        '      here are not directly comparable without those controls.',
        'NOTE: p-values and CIs are unreliable with small samples (n < 30).',
        '      Use cross-validated AUC as the primary summary metric.',
        '',
    ]

    if n_total < 4:
        lines.append('ERROR: Too few subjects for logistic regression (need at least 4).')
        _write_txt(out_dir, 'auc_regression.txt', lines)
        print("  Skipped statistics — too few subjects.")
        return

    # Build outcome vector (ADHD=1, TDC=0)
    y = np.array([1 if s['group'] == 'ADHD' else 0 for s in subjects])

    # Header row for results table
    lines.append(
        f"{'measure':<12} {'band':<7} {'homer':<6} {'predictor':<12} "
        f"{'coef':>8} {'p':>8} {'CI_lo':>8} {'CI_hi':>8} {'cv_AUC':>8} {'n':>5}"
    )
    lines.append('-' * 90)

    for homer in ['0', '1']:
        for m in MEASURES:
            for b in BANDS:
                key = (m, b)

                # Collect per-subject scalars
                rows_data = []
                for s in subjects:
                    feat_dict = _feat(s, pipeline)
                    if feat_dict is None or key not in feat_dict:
                        continue
                    feat = feat_dict[key]
                    rows_data.append({
                        'auc':      feat[f'auc_b{homer}'],
                        'slope':    feat[f'slope_b{homer}'],
                        'kurtosis': feat[f'kurtosis_b{homer}'],
                        'idx':      subjects.index(s),
                    })

                if len(rows_data) < 4:
                    continue

                idxs  = [r['idx'] for r in rows_data]
                y_sub = y[idxs]

                # Standardize predictors
                raw_X = np.column_stack([
                    [r['auc']      for r in rows_data],
                    [r['slope']    for r in rows_data],
                    [r['kurtosis'] for r in rows_data],
                ])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    X_std = (raw_X - raw_X.mean(axis=0)) / (raw_X.std(axis=0) + 1e-12)

                X_full     = X_std
                X_sm       = sm.add_constant(X_full)
                pred_names = ['const', 'auc', 'slope', 'kurtosis']

                # Cross-validated AUC
                cv_auc = None
                if sklearn_available and len(np.unique(y_sub)) == 2:
                    try:
                        clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
                        cv  = min(5, int(y_sub.sum()), int((1 - y_sub).sum()))
                        if cv >= 2:
                            scores  = cross_val_score(clf, X_full, y_sub,
                                                      cv=cv, scoring='roc_auc')
                            cv_auc  = float(np.mean(scores))
                    except Exception:
                        pass

                # Statsmodels logistic regression
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        result = sm.Logit(y_sub, X_sm).fit(disp=False, maxiter=200)

                    cv_str = f'{cv_auc:.3f}' if cv_auc is not None else '   N/A'
                    n_str  = str(len(rows_data))

                    for pred in ['auc', 'slope', 'kurtosis']:
                        i    = pred_names.index(pred)
                        coef = result.params[i]
                        pval = result.pvalues[i]
                        ci   = result.conf_int()
                        ci_lo, ci_hi = ci[i, 0], ci[i, 1]

                        sig = ''
                        if pval < 0.001: sig = '***'
                        elif pval < 0.01: sig = '** '
                        elif pval < 0.05: sig = '*  '
                        else:             sig = '   '

                        lines.append(
                            f"{m:<12} {b:<7} B{homer:<5} {pred:<12} "
                            f"{coef:>8.3f} {pval:>8.3f}{sig} "
                            f"{ci_lo:>8.3f} {ci_hi:>8.3f} "
                            f"{cv_str:>8} {n_str:>5}"
                        )

                    lines.append('')  # blank row between measure/band blocks

                except Exception as e:
                    lines.append(f"{m:<12} {b:<7} B{homer:<5} [regression failed: {e}]")

    _write_txt(out_dir, 'auc_regression.txt', lines)
    print(f"  Saved auc_regression.txt")


def _parse_run_log(subject_id):
    """
    Parse a subject's run.log and return a dict of preprocessing stats.
    Returns None if log not found or STATUS != complete.
    """
    log_path = os.path.join(config.RESULTS_ROOT, 'subjects', subject_id, 'run.log')
    if not os.path.exists(log_path):
        return None

    text = open(log_path, encoding='utf-8', errors='replace').read()

    # STATUS
    status_m = re.search(r'STATUS:\s*(\S.*)', text)
    status   = status_m.group(1).strip() if status_m else 'unknown'

    # RANSAC bad channels  →  "RANSAC bad channels: ['Fp1', 'F7']"
    ransac_m   = re.search(r'RANSAC bad channels:\s*(\[.*?\])', text)
    try:
        bad_chs = eval(ransac_m.group(1)) if ransac_m else []
    except Exception:
        bad_chs = []

    # ICA excluded  →  "ICLabel excluded 3 components: [0, 1, 4]"
    #              OR  "mne-icalabel not installed — using hardcoded ica.exclude=[0,1]"
    ica_m = re.search(r'ICLabel excluded (\d+) components', text)
    if ica_m:
        ica_n      = int(ica_m.group(1))
        ica_method = 'ICLabel'
    elif 'hardcoded ica.exclude' in text:
        ica_n      = 2
        ica_method = 'hardcoded'
    else:
        ica_n      = None
        ica_method = 'unknown'

    # Epochs  →  "Epochs: 80 -> 66 after AutoReject (14 dropped)"
    ep_m = re.search(r'Epochs:\s*(\d+)\s*-[->]\s*(\d+)\s*after AutoReject\s*\((\d+) dropped\)', text)
    if ep_m:
        epochs_before  = int(ep_m.group(1))
        epochs_after   = int(ep_m.group(2))
        epochs_dropped = int(ep_m.group(3))
    else:
        epochs_before = epochs_after = epochs_dropped = None

    # KPSS  →  "KPSS stationarity: 4/1188 tests failed (0.3%)"
    kpss_m    = re.search(r'KPSS stationarity:\s*(\d+)/(\d+) tests failed \(([\d.]+)%\)', text)
    kpss_fail = float(kpss_m.group(3)) if kpss_m else None

    # CSD  →  "CSD transform applied."
    has_csd = 'CSD transform applied' in text

    return {
        'status':         status,
        'bad_chs':        bad_chs,
        'n_bad':          len(bad_chs),
        'ica_n':          ica_n,
        'ica_method':     ica_method,
        'epochs_before':  epochs_before,
        'epochs_after':   epochs_after,
        'epochs_dropped': epochs_dropped,
        'kpss_fail_pct':  kpss_fail,
        'has_csd':        has_csd,
    }


def generate_preprocessing_summary(subjects, skipped, out_dir):
    """
    Parse run.log for every subject and write overview/preprocessing_summary.txt.
    """
    now    = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    n_adhd = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc  = sum(1 for s in subjects if s['group'] == 'TDC')

    # Gather per-subject stats
    rows = []
    for s in subjects:
        log = _parse_run_log(s['subject_id'])
        rows.append({'subject_id': s['subject_id'], 'group': s['group'],
                     'has_sl_feat': s['features_sl'] is not None,
                     **(log or {})})

    failed = [sid for sid in skipped if _parse_run_log(sid) is not None
              and (_parse_run_log(sid) or {}).get('status', '') != 'complete']
    not_run = [sid for sid in skipped if _parse_run_log(sid) is None]

    def _stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return 'N/A'
        return f'{np.mean(vals):.1f} ± {np.std(vals):.1f}'

    def _group_rows(group):
        return [r for r in rows if r.get('group') == group]

    lines = [
        'ADHD EEG TDA Pipeline — Preprocessing & Group Summary',
        f'Generated: {now}',
        '=' * 60,
        '',
        'COHORT',
        '-' * 60,
        f'  Completed subjects : {len(subjects)}  (ADHD={n_adhd}, TDC={n_tdc})',
        f'  Skipped (not run)  : {len(not_run)}',
        f'  Failed             : {len(failed)}',
    ]

    if failed:
        for sid in failed:
            log = _parse_run_log(sid)
            lines.append(f'    {sid}: {log.get("status", "?")}')

    lines += [
        '',
        'PREPROCESSING — GROUP AVERAGES',
        '-' * 60,
        f'  {"Metric":<28} {"ADHD":>12} {"TDC":>12}',
        f'  {"-"*28} {"-"*12} {"-"*12}',
    ]

    metrics = [
        ('Epochs before AutoReject', 'epochs_before'),
        ('Epochs after AutoReject',  'epochs_after'),
        ('Epochs dropped',           'epochs_dropped'),
        ('RANSAC bad channels',      'n_bad'),
        ('ICA components excluded',  'ica_n'),
        ('KPSS fail rate (%)',        'kpss_fail_pct'),
    ]
    for label, key in metrics:
        adhd_vals = [r.get(key) for r in _group_rows('ADHD')]
        tdc_vals  = [r.get(key) for r in _group_rows('TDC')]
        lines.append(f'  {label:<28} {_stats(adhd_vals):>12} {_stats(tdc_vals):>12}')

    # Surface Laplacian availability
    n_sl_adhd = sum(1 for r in _group_rows('ADHD') if r.get('has_sl_feat'))
    n_sl_tdc  = sum(1 for r in _group_rows('TDC')  if r.get('has_sl_feat'))
    lines += [
        '',
        f'  {"Surface Laplacian features":<28} {f"{n_sl_adhd}/{n_adhd}":>12} {f"{n_sl_tdc}/{n_tdc}":>12}',
    ]

    # Per-subject table
    lines += [
        '',
        'PREPROCESSING — PER SUBJECT',
        '-' * 60,
        f'  {"Subject":<10} {"Group":<6} {"Before":>7} {"After":>6} '
        f'{"Drop":>5} {"BadCh":>6} {"ICA":>4} {"KPSS%":>6} {"SL":>4} {"Status"}',
        f'  {"-"*10} {"-"*6} {"-"*7} {"-"*6} {"-"*5} {"-"*6} {"-"*4} {"-"*6} {"-"*4} {"-"*12}',
    ]

    for r in sorted(rows, key=lambda x: (x['group'], x['subject_id'])):
        sid    = r['subject_id']
        grp    = r.get('group', '?')
        before = str(r['epochs_before'])  if r.get('epochs_before')  is not None else '-'
        after  = str(r['epochs_after'])   if r.get('epochs_after')   is not None else '-'
        drop   = str(r['epochs_dropped']) if r.get('epochs_dropped') is not None else '-'
        bad    = str(r['n_bad'])          if r.get('n_bad')          is not None else '-'
        ica    = str(r['ica_n'])          if r.get('ica_n')          is not None else '-'
        kpss   = f"{r['kpss_fail_pct']:.1f}" if r.get('kpss_fail_pct') is not None else '-'
        csd    = 'Y' if r.get('has_sl_feat') else '-'
        status = r.get('status', 'unknown')
        lines.append(
            f'  {sid:<10} {grp:<6} {before:>7} {after:>6} {drop:>5} '
            f'{bad:>6} {ica:>4} {kpss:>6} {csd:>4}  {status}'
        )

    # Not-run subjects
    if not_run:
        lines += ['', f'  Not yet run ({len(not_run)}): {", ".join(sorted(not_run))}']

    lines += [
        '',
        '=' * 60,
        'PIPELINE CONFIGURATION',
        '-' * 60,
        f'  Sample rate         : {config.SFREQ} Hz',
        f'  Epoch duration      : {config.EPOCH_DURATION} s',
        f'  Min clean epochs    : {config.MIN_CLEAN_EPOCHS}',
        f'  ICA components      : {config.ICA_N_COMPONENTS}',
        f'  ICA method          : {config.ICA_METHOD} (extended={config.ICA_EXTENDED})',
        f'  ICLabel threshold   : {config.ICLABEL_THRESHOLD}',
        f'  Connectivity        : {", ".join(config.CONN_MEASURES)}',
        f'  Frequency bands     : {", ".join(f"{k} {v}" for k, v in config.FREQ_BANDS.items())}',
        f'  Graph density       : {config.GRAPH_DENSITY}',
        f'  TDA filter steps    : {config.N_FILT_STEPS}',
        '',
        'See no_filter/auc_regression.txt and surface_laplacian/auc_regression.txt for statistical results.',
    ]

    _write_txt(out_dir, 'preprocessing_summary.txt', lines)
    print('  Saved preprocessing_summary.txt')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — Graph theory metrics
# ══════════════════════════════════════════════════════════════════════════════

GRAPH_METRICS = ['strength', 'clustering', 'cpl', 'betweenness', 'efficiency', 'modularity']
GRAPH_LABELS  = {
    'strength':    'Strength',
    'clustering':  'Clustering',
    'cpl':         'Char. Path Length',
    'betweenness': 'Betweenness',
    'efficiency':  'Global Efficiency',
    'modularity':  'Modularity',
}
GRAPH_NOTES = {
    'strength':    'higher = stronger connections',
    'clustering':  'higher = more local clustering',
    'cpl':         'higher = longer paths (less efficient)',
    'betweenness': 'higher = more hub-like nodes',
    'efficiency':  'higher = more parallel paths',
    'modularity':  'higher = more distinct communities',
}


def _load_graph_metrics(subjects, pipeline='no_filter'):
    """
    Load graph_results.pkl (no_filter) or graph_results_csd.pkl (surface_laplacian) from each
    subject's cache. Stores result in s['graph'].
    """
    fname = 'graph_results_csd.pkl' if pipeline == 'surface_laplacian' else 'graph_results.pkl'
    for s in subjects:
        sid        = s['subject_id']
        cache_path = os.path.join(config.RESULTS_ROOT, 'subjects', sid, '.cache', fname)
        try:
            with open(cache_path, 'rb') as f:
                s['graph'] = pickle.load(f)
        except Exception:
            s['graph'] = None
    return subjects


def plot_graph_metrics_group(subjects, out_dir, pipeline='no_filter'):
    """
    One figure per connectivity measure (3 total).
    Layout: 6 rows (metrics) x 5 columns (bands).
    Per-row y-scale. ADHD vs TDC boxplot + jitter.
    """
    pipeline_label = 'No Filter' if pipeline == 'no_filter' else 'Surface Laplacian'
    subjects = _load_graph_metrics(subjects, pipeline=pipeline)
    n_adhd   = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc    = sum(1 for s in subjects if s['group'] == 'TDC')
    rng      = np.random.default_rng(42)

    os.makedirs(out_dir, exist_ok=True)

    for m in MEASURES:
        fig, axes = plt.subplots(6, 5, figsize=(18, 20))
        fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.94, bottom=0.04,
                            left=0.08, right=0.98)

        for row, metric in enumerate(GRAPH_METRICS):
            # Per-row y limits
            row_vals = []
            for b in BANDS:
                for grp in ['ADHD', 'TDC']:
                    for s in subjects:
                        if s.get('graph') and (m, b) in s['graph']:
                            v = s['graph'][(m, b)].get(metric)
                            if v is not None:
                                row_vals.append(v)
            if row_vals:
                rmin = min(row_vals)
                rmax = max(row_vals)
                pad  = (rmax - rmin) * 0.18 if rmax > rmin else 0.1
                ylim = (rmin - pad, rmax + pad)
            else:
                ylim = (0, 1)

            for col, b in enumerate(BANDS):
                ax = axes[row, col]

                adhd_vals = [s['graph'][(m, b)][metric]
                             for s in subjects
                             if s.get('graph') and (m, b) in s['graph']
                             and metric in s['graph'][(m, b)]
                             and s['group'] == 'ADHD']
                tdc_vals  = [s['graph'][(m, b)][metric]
                             for s in subjects
                             if s.get('graph') and (m, b) in s['graph']
                             and metric in s['graph'][(m, b)]
                             and s['group'] == 'TDC']

                for vals, x_pos, color in [(adhd_vals, 1, ADHD_COLOR),
                                           (tdc_vals,  2, TDC_COLOR)]:
                    if not vals:
                        continue
                    bp = ax.boxplot([vals], positions=[x_pos], patch_artist=True,
                                   widths=0.4, showfliers=False,
                                   medianprops=dict(color='#333333', linewidth=1.5))
                    bp['boxes'][0].set_facecolor(color)
                    bp['boxes'][0].set_alpha(0.5)
                    jitter = rng.uniform(-0.12, 0.12, len(vals))
                    ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                               color=color, s=15, alpha=0.65, zorder=3)

                ax.set_xlim(0.3, 2.7)
                ax.set_ylim(*ylim)
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['ADHD', 'TDC'], fontsize=7)
                ax.tick_params(labelsize=7)
                ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

                if row == 0:
                    ax.set_title(BAND_LABELS[b], fontsize=11, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f'{GRAPH_LABELS[metric]}\n({GRAPH_NOTES[metric]})',
                                  fontsize=7)

        fig.suptitle(
            f'Graph Theory Metrics — {MEASURE_LABELS[m]}  |  {pipeline_label}\n'
            f'ADHD (n={n_adhd}) vs TDC (n={n_tdc})  |  '
            f'Density threshold = {config.GRAPH_DENSITY}',
            fontsize=12,
        )
        fname_tag = '_sl' if pipeline == 'surface_laplacian' else ''
        fname = f'graph_metrics_{m}{fname_tag}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_graph_metrics_combined(subjects, out_dir):
    """
    Combined no-filter + surface laplacian graph theory metrics. One figure per measure (3 total).
    Layout: 12 rows (6 metrics × 2 pipelines interleaved) × 5 bands.
    Grey background = No Filter, blue tint = Surface Laplacian. Shared y-axis per metric row pair.
    """
    BG_NF = (0.95, 0.95, 0.95)
    BG_SL = (0.90, 0.94, 1.00)

    for s in subjects:
        sid       = s['subject_id']
        cache_dir = os.path.join(config.RESULTS_ROOT, 'subjects', sid, '.cache')
        for key, fname in [('graph_nf', 'graph_results.pkl'),
                           ('graph_sl', 'graph_results_csd.pkl')]:
            try:
                with open(os.path.join(cache_dir, fname), 'rb') as f:
                    s[key] = pickle.load(f)
            except Exception:
                s[key] = None

    has_nf = any(s.get('graph_nf') for s in subjects)
    has_sl = any(s.get('graph_sl') for s in subjects)
    if not has_nf or not has_sl:
        print("  Skipping graph metrics combined — need both no-filter and surface laplacian cache files.")
        return

    n_adhd = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc  = sum(1 for s in subjects if s['group'] == 'TDC')
    rng    = np.random.default_rng(42)
    os.makedirs(out_dir, exist_ok=True)

    for m in MEASURES:
        fig, axes = plt.subplots(12, 5, figsize=(18, 38))
        fig.subplots_adjust(hspace=0.32, wspace=0.35, top=0.97, bottom=0.01,
                            left=0.10, right=0.98)

        for metric_idx, metric in enumerate(GRAPH_METRICS):
            nf_row = metric_idx * 2
            sl_row = metric_idx * 2 + 1

            # Shared y-limits across both pipelines for this metric
            all_vals = []
            for pk in ['graph_nf', 'graph_sl']:
                for b in BANDS:
                    for s in subjects:
                        if s.get(pk) and (m, b) in s[pk]:
                            v = s[pk][(m, b)].get(metric)
                            if v is not None:
                                all_vals.append(v)
            if all_vals:
                rmin = min(all_vals)
                rmax = max(all_vals)
                pad  = (rmax - rmin) * 0.18 if rmax > rmin else 0.1
                ylim = (rmin - pad, rmax + pad)
            else:
                ylim = (0, 1)

            for col, b in enumerate(BANDS):
                for row, (pk, bg, lbl) in enumerate([
                    ('graph_nf', BG_NF, 'No Filter'),
                    ('graph_sl', BG_SL, 'Surface Laplacian'),
                ], start=nf_row):
                    ax = axes[row, col]
                    ax.set_facecolor(bg)

                    adhd_vals = [s[pk][(m, b)][metric]
                                 for s in subjects
                                 if s.get(pk) and (m, b) in s[pk]
                                 and metric in s[pk][(m, b)]
                                 and s['group'] == 'ADHD']
                    tdc_vals  = [s[pk][(m, b)][metric]
                                 for s in subjects
                                 if s.get(pk) and (m, b) in s[pk]
                                 and metric in s[pk][(m, b)]
                                 and s['group'] == 'TDC']

                    for vals, x_pos, color in [(adhd_vals, 1, ADHD_COLOR),
                                               (tdc_vals,  2, TDC_COLOR)]:
                        if not vals:
                            continue
                        bp = ax.boxplot([vals], positions=[x_pos], patch_artist=True,
                                       widths=0.4, showfliers=False,
                                       medianprops=dict(color='#333333', linewidth=1.5))
                        bp['boxes'][0].set_facecolor(color)
                        bp['boxes'][0].set_alpha(0.5)
                        jitter = rng.uniform(-0.12, 0.12, len(vals))
                        ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                                   color=color, s=10, alpha=0.65, zorder=3)

                    ax.set_xlim(0.3, 2.7)
                    ax.set_ylim(*ylim)
                    ax.set_xticks([1, 2])
                    ax.set_xticklabels(['A', 'T'], fontsize=5)
                    ax.tick_params(labelsize=5)
                    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

                    if row == 0 and col == 0:
                        ax.set_title(BAND_LABELS[b], fontsize=10, fontweight='bold')
                    elif row == 0:
                        ax.set_title(BAND_LABELS[b], fontsize=10, fontweight='bold')
                    if col == 0:
                        ax.set_ylabel(f'{GRAPH_LABELS[metric]}\n[{lbl}]', fontsize=6)

        fig.suptitle(
            f'Graph Theory Metrics — {MEASURE_LABELS[m]}  |  Raw (grey) vs Surface Laplacian (blue)\n'
            f'ADHD (n={n_adhd}) vs TDC (n={n_tdc})  |  '
            f'Density = {config.GRAPH_DENSITY}  |  Shared y-axis per metric',
            fontsize=11,
        )
        fname = f'graph_metrics_{m}_combined.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — PCA / UMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_umap(subjects, out_dir, pipeline='no_filter'):
    """
    Reduce all TDA features to 2D via PCA (always) and UMAP (if installed).
    Features: auc_b0, slope_b0, kurtosis_b0, auc_b1, slope_b1, kurtosis_b1
              for all 15 (measure x band) combinations = 90 features per subject.
    pipeline: 'no_filter' or 'surface_laplacian'
    """
    pipeline_label = 'No Filter' if pipeline == 'no_filter' else 'Surface Laplacian'
    fname_tag      = '' if pipeline == 'no_filter' else '_sl'

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except ImportError:
        print("  Skipping PCA/UMAP — scikit-learn not installed.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Build feature matrix
    feature_names = []
    for m in MEASURES:
        for b in BANDS:
            for feat in ['auc_b0', 'slope_b0', 'kurtosis_b0',
                         'auc_b1', 'slope_b1', 'kurtosis_b1']:
                feature_names.append(f'{m[:3]}_{b[:3]}_{feat}')

    rows, groups, sids = [], [], []
    for s in subjects:
        feat_dict = _feat(s, pipeline)
        if feat_dict is None:
            continue
        row = []
        for m in MEASURES:
            for b in BANDS:
                key  = (m, b)
                feat = feat_dict.get(key, {})
                for f in ['auc_b0', 'slope_b0', 'kurtosis_b0',
                          'auc_b1', 'slope_b1', 'kurtosis_b1']:
                    row.append(float(feat.get(f, 0.0)))
        rows.append(row)
        groups.append(s['group'])
        sids.append(s['subject_id'])

    if len(rows) < 4:
        print("  Skipping PCA/UMAP — not enough subjects.")
        return

    X      = np.array(rows)
    y      = np.array([1 if g == 'ADHD' else 0 for g in groups])
    colors = [ADHD_COLOR if g == 'ADHD' else TDC_COLOR for g in groups]

    # Standardize
    X_std = StandardScaler().fit_transform(X)

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca      = PCA(n_components=min(10, X_std.shape[1]))
    X_pca    = pca.fit_transform(X_std)
    var_exp  = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.10)

    # PC1 vs PC2
    ax = axes[0]
    for grp, color, label in [('ADHD', ADHD_COLOR, f'ADHD (n={sum(y==1)})'),
                                ('TDC',  TDC_COLOR,  f'TDC  (n={sum(y==0)})')]:
        mask = np.array(groups) == grp
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, s=50, alpha=0.75, label=label,
                   edgecolors='white', linewidths=0.4)
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}% variance)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}% variance)', fontsize=10)
    ax.set_title('PCA — PC1 vs PC2', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Scree plot
    ax = axes[1]
    n_show = min(10, len(var_exp))
    ax.bar(range(1, n_show + 1), var_exp[:n_show] * 100,
           color='#5b8dd9', alpha=0.8, edgecolor='white')
    ax.plot(range(1, n_show + 1), np.cumsum(var_exp[:n_show]) * 100,
            'o-', color='#e74c3c', linewidth=1.8, markersize=5, label='Cumulative')
    ax.set_xlabel('Principal Component', fontsize=10)
    ax.set_ylabel('Variance Explained (%)', fontsize=10)
    ax.set_title('Scree Plot', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle(
        f'PCA of TDA Features — ADHD vs TDC  |  {pipeline_label}\n'
        f'90 features (auc/slope/kurtosis × B0/B1 × 3 measures × 5 bands)',
        fontsize=12,
    )
    fig.savefig(os.path.join(out_dir, f'pca{fname_tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved pca{fname_tag}.png")

    # ── Top PC loadings ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.4, top=0.88, bottom=0.25)
    for pc_idx, ax in enumerate(axes):
        loadings = pca.components_[pc_idx]
        top_idx  = np.argsort(np.abs(loadings))[-15:][::-1]
        top_vals = loadings[top_idx]
        top_names = [feature_names[i] for i in top_idx]
        bar_colors = [ADHD_COLOR if v > 0 else TDC_COLOR for v in top_vals]
        ax.barh(range(len(top_vals)), top_vals[::-1], color=bar_colors[::-1], alpha=0.8)
        ax.set_yticks(range(len(top_vals)))
        ax.set_yticklabels(top_names[::-1], fontsize=7)
        ax.set_xlabel('Loading', fontsize=9)
        ax.set_title(f'PC{pc_idx+1} Top 15 Loadings ({var_exp[pc_idx]*100:.1f}% var)',
                     fontsize=10)
        ax.axvline(0, color='#333333', linewidth=0.8)
        ax.grid(True, axis='x', linewidth=0.3, alpha=0.4)
    fig.suptitle(f'PCA Feature Loadings — {pipeline_label}', fontsize=11)
    fig.savefig(os.path.join(out_dir, f'pca_loadings{fname_tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved pca_loadings{fname_tag}.png")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    try:
        import umap
        reducer  = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap   = reducer.fit_transform(X_std)

        fig, ax = plt.subplots(figsize=(8, 6))
        for grp, color, label in [('ADHD', ADHD_COLOR, f'ADHD (n={sum(y==1)})'),
                                    ('TDC',  TDC_COLOR,  f'TDC  (n={sum(y==0)})')]:
            mask = np.array(groups) == grp
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=color, s=55, alpha=0.75, label=label,
                       edgecolors='white', linewidths=0.4)
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.set_title(
            f'UMAP of TDA Features — ADHD vs TDC\n'
            f'n_neighbors=15, min_dist=0.1',
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        fig.savefig(os.path.join(out_dir, f'umap{fname_tag}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved umap{fname_tag}.png")
    except ImportError:
        print("  Skipping UMAP — umap-learn not installed. Run: pip install umap-learn")


def plot_pca_combined(subjects, out_dir):
    """
    Combined no-filter + surface laplacian PCA figures.
    pca_combined.png      — 2×2: [no_filter scatter, no_filter scree; sl scatter, sl scree]
    pca_loadings_combined.png — 2×2: [no_filter PC1, no_filter PC2; sl PC1, sl PC2]
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except ImportError:
        print("  Skipping PCA combined — scikit-learn not installed.")
        return

    os.makedirs(out_dir, exist_ok=True)

    BG_NF = (0.95, 0.95, 0.95)
    BG_SL = (0.90, 0.94, 1.00)

    feature_names = []
    for m in MEASURES:
        for b in BANDS:
            for feat in ['auc_b0', 'slope_b0', 'kurtosis_b0',
                         'auc_b1', 'slope_b1', 'kurtosis_b1']:
                feature_names.append(f'{m[:3]}_{b[:3]}_{feat}')

    def _build_matrix(pipeline):
        rows, groups_ = [], []
        for s in subjects:
            feat_dict = _feat(s, pipeline)
            if feat_dict is None:
                continue
            row = []
            for m in MEASURES:
                for b in BANDS:
                    feat = feat_dict.get((m, b), {})
                    for f in ['auc_b0', 'slope_b0', 'kurtosis_b0',
                              'auc_b1', 'slope_b1', 'kurtosis_b1']:
                        row.append(float(feat.get(f, 0.0)))
            rows.append(row)
            groups_.append(s['group'])
        return np.array(rows) if rows else None, groups_

    X_nf, grp_nf = _build_matrix('no_filter')
    X_sl, grp_sl = _build_matrix('surface_laplacian')

    if X_nf is None or X_sl is None or len(X_nf) < 4 or len(X_sl) < 4:
        print("  Skipping PCA combined — not enough subjects with both pipelines.")
        return

    results = {}
    for tag, X, grp in [('no_filter', X_nf, grp_nf), ('surface_laplacian', X_sl, grp_sl)]:
        X_std = StandardScaler().fit_transform(X)
        pca   = PCA(n_components=min(10, X_std.shape[1]))
        X_pca = pca.fit_transform(X_std)
        results[tag] = dict(X_pca=X_pca, pca=pca, groups=grp,
                            var=pca.explained_variance_ratio_)

    labels_map = {'no_filter': 'No Filter', 'surface_laplacian': 'Surface Laplacian'}
    bg_map     = {'no_filter': BG_NF, 'surface_laplacian': BG_SL}

    # ── pca_combined.png ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(hspace=0.40, wspace=0.35, top=0.91, bottom=0.07,
                        left=0.08, right=0.97)

    for row_idx, tag in enumerate(['no_filter', 'surface_laplacian']):
        r      = results[tag]
        grp    = r['groups']
        X_pca  = r['X_pca']
        var    = r['var']
        pl_lbl = labels_map[tag]
        bg     = bg_map[tag]

        # Scatter
        ax = axes[row_idx, 0]
        ax.set_facecolor(bg)
        n_adhd = sum(1 for g in grp if g == 'ADHD')
        n_tdc  = sum(1 for g in grp if g == 'TDC')
        for grp_name, color, lbl in [('ADHD', ADHD_COLOR, f'ADHD (n={n_adhd})'),
                                      ('TDC',  TDC_COLOR,  f'TDC  (n={n_tdc})')]:
            mask = np.array(grp) == grp_name
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=color, s=45, alpha=0.75, label=lbl,
                       edgecolors='white', linewidths=0.4)
        ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=9)
        ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=9)
        ax.set_title(f'PCA — {pl_lbl}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.4)

        # Scree
        ax = axes[row_idx, 1]
        ax.set_facecolor(bg)
        n_show = min(10, len(var))
        ax.bar(range(1, n_show + 1), var[:n_show] * 100,
               color='#5b8dd9', alpha=0.8, edgecolor='white')
        ax.plot(range(1, n_show + 1), np.cumsum(var[:n_show]) * 100,
                'o-', color='#e74c3c', linewidth=1.8, markersize=5, label='Cumulative')
        ax.set_xlabel('PC', fontsize=9)
        ax.set_ylabel('Variance Explained (%)', fontsize=9)
        ax.set_title(f'Scree — {pl_lbl}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle(
        'PCA of TDA Features — Raw (grey) vs Surface Laplacian (blue)\n'
        '90 features (auc/slope/kurtosis × B0/B1 × 3 measures × 5 bands)',
        fontsize=12,
    )
    fig.savefig(os.path.join(out_dir, 'pca_combined.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved pca_combined.png")

    # ── pca_loadings_combined.png ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.50, wspace=0.45, top=0.91, bottom=0.20,
                        left=0.18, right=0.97)

    for row_idx, tag in enumerate(['no_filter', 'surface_laplacian']):
        r      = results[tag]
        pca    = r['pca']
        var    = r['var']
        pl_lbl = labels_map[tag]
        bg     = bg_map[tag]
        for pc_idx in range(2):
            ax       = axes[row_idx, pc_idx]
            ax.set_facecolor(bg)
            loadings = pca.components_[pc_idx]
            top_idx  = np.argsort(np.abs(loadings))[-15:][::-1]
            top_vals = loadings[top_idx]
            top_names = [feature_names[i] for i in top_idx]
            bar_colors = [ADHD_COLOR if v > 0 else TDC_COLOR for v in top_vals]
            ax.barh(range(len(top_vals)), top_vals[::-1], color=bar_colors[::-1], alpha=0.8)
            ax.set_yticks(range(len(top_vals)))
            ax.set_yticklabels(top_names[::-1], fontsize=6)
            ax.set_xlabel('Loading', fontsize=8)
            ax.set_title(
                f'PC{pc_idx+1} ({var[pc_idx]*100:.1f}% var) — {pl_lbl}',
                fontsize=9, fontweight='bold',
            )
            ax.axvline(0, color='#333333', linewidth=0.8)
            ax.grid(True, axis='x', linewidth=0.3, alpha=0.4)

    fig.suptitle(
        'PCA Feature Loadings — Top 15 per PC\nRaw (grey) vs Surface Laplacian (blue)',
        fontsize=12,
    )
    fig.savefig(os.path.join(out_dir, 'pca_loadings_combined.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved pca_loadings_combined.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — Density sweep (graph metrics vs threshold)
# ══════════════════════════════════════════════════════════════════════════════

def plot_density_sweep_group(subjects, out_dir, pipeline='no_filter'):
    """
    For each connectivity measure: 6 metrics × 5 bands grid.
    Each panel shows group mean ± 95% CI of the metric across density thresholds.
    Shows whether ADHD/TDC differences are robust across thresholds or threshold-dependent.
    pipeline: 'no_filter' or 'surface_laplacian'
    """
    pipeline_label = 'No Filter' if pipeline == 'no_filter' else 'Surface Laplacian'
    fname          = 'density_sweep_csd.pkl' if pipeline == 'surface_laplacian' else 'density_sweep.pkl'

    os.makedirs(out_dir, exist_ok=True)

    # Load density sweep for each subject
    sweeps    = {}   # sid -> (sweep_dict, densities)
    densities = None
    for s in subjects:
        sid        = s['subject_id']
        cache_path = os.path.join(config.RESULTS_ROOT, 'subjects', sid,
                                  '.cache', fname)
        try:
            with open(cache_path, 'rb') as f:
                sweep, dens = pickle.load(f)
            sweeps[sid]  = sweep
            densities    = dens
        except Exception:
            pass

    if not sweeps or densities is None:
        print("  Skipping density sweep — no data found.")
        return

    n_adhd = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc  = sum(1 for s in subjects if s['group'] == 'TDC')

    for m in MEASURES:
        fig, axes = plt.subplots(6, 5, figsize=(18, 20))
        fig.subplots_adjust(hspace=0.50, wspace=0.35, top=0.94, bottom=0.04,
                            left=0.09, right=0.98)

        for row, metric in enumerate(GRAPH_METRICS):
            for col, b in enumerate(BANDS):
                ax  = axes[row, col]
                key = (m, b)

                for grp, color, label in [('ADHD', ADHD_COLOR, 'ADHD'),
                                           ('TDC',  TDC_COLOR,  'TDC')]:
                    curves = []
                    for s in subjects:
                        if s['group'] != grp:
                            continue
                        sw = sweeps.get(s['subject_id'])
                        if sw and key in sw and metric in sw[key]:
                            curves.append(sw[key][metric])

                    if not curves:
                        continue

                    arr  = np.array(curves)   # (n_subjects, n_densities)
                    mean = arr.mean(axis=0)
                    n    = arr.shape[0]
                    if n > 1:
                        ci = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(n)
                    else:
                        ci = np.zeros_like(mean)

                    ax.plot(densities, mean, color=color, linewidth=1.8, label=label)
                    ax.fill_between(densities, mean - ci, mean + ci,
                                    color=color, alpha=0.18)

                # Mark our chosen analysis density
                ax.axvline(config.GRAPH_DENSITY, color='#888888',
                           linewidth=0.8, linestyle='--', alpha=0.7)

                ax.set_xlim(densities[0], densities[-1])
                ax.tick_params(labelsize=7)
                ax.grid(True, linewidth=0.3, alpha=0.4)

                if row == 0:
                    ax.set_title(BAND_LABELS[b], fontsize=11, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f'{GRAPH_LABELS[metric]}', fontsize=8)
                if row == 5:
                    ax.set_xlabel('Density', fontsize=7)

        axes[0, 0].legend(fontsize=7, loc='best', framealpha=0.8)

        fig.suptitle(
            f'Graph Metrics vs Density Threshold — {MEASURE_LABELS[m]}  |  {pipeline_label}\n'
            f'ADHD (n={n_adhd}) vs TDC (n={n_tdc})  |  '
            f'Mean ± 95% CI  |  Dashed line = analysis threshold ({config.GRAPH_DENSITY})',
            fontsize=12,
        )
        fname_tag = '_sl' if pipeline == 'surface_laplacian' else ''
        fname = f'density_sweep_{m}{fname_tag}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_density_sweep_combined(subjects, out_dir):
    """
    Combined no-filter + surface laplacian density sweep. One figure per measure (3 total).
    Layout: 12 rows (6 metrics × 2 pipelines interleaved) × 5 bands.
    Grey background = No Filter, blue tint = Surface Laplacian. Shared y-axis per metric row pair.
    """
    BG_NF = (0.95, 0.95, 0.95)
    BG_SL = (0.90, 0.94, 1.00)

    sweeps    = {'no_filter': {}, 'surface_laplacian': {}}
    densities = None

    for s in subjects:
        sid       = s['subject_id']
        cache_dir = os.path.join(config.RESULTS_ROOT, 'subjects', sid, '.cache')
        for tag, fname in [('no_filter', 'density_sweep.pkl'), ('surface_laplacian', 'density_sweep_csd.pkl')]:
            try:
                with open(os.path.join(cache_dir, fname), 'rb') as f:
                    sweep, dens = pickle.load(f)
                sweeps[tag][sid] = sweep
                densities        = dens
            except Exception:
                pass

    has_nf = bool(sweeps['no_filter'])
    has_sl = bool(sweeps['surface_laplacian'])
    if not has_nf or not has_sl or densities is None:
        print("  Skipping density sweep combined — need both no-filter and surface laplacian cache files.")
        return

    n_adhd = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc  = sum(1 for s in subjects if s['group'] == 'TDC')
    os.makedirs(out_dir, exist_ok=True)

    for m in MEASURES:
        fig, axes = plt.subplots(12, 5, figsize=(18, 38))
        fig.subplots_adjust(hspace=0.35, wspace=0.35, top=0.97, bottom=0.01,
                            left=0.10, right=0.98)

        for metric_idx, metric in enumerate(GRAPH_METRICS):
            nf_row = metric_idx * 2
            sl_row = metric_idx * 2 + 1
            key = None   # will be set per band

            # Compute shared y-limits across both pipelines for this metric
            all_vals = []
            for tag in ['no_filter', 'surface_laplacian']:
                for b in BANDS:
                    k = (m, b)
                    for sw in sweeps[tag].values():
                        if sw and k in sw and metric in sw[k]:
                            all_vals.extend(sw[k][metric])
            if all_vals:
                vmin_ = min(all_vals)
                vmax_ = max(all_vals)
                pad   = (vmax_ - vmin_) * 0.15 if vmax_ > vmin_ else 0.1
                ylim  = (vmin_ - pad, vmax_ + pad)
            else:
                ylim = (0, 1)

            for col, b in enumerate(BANDS):
                k = (m, b)
                for row, (tag, bg, lbl) in enumerate([
                    ('no_filter', BG_NF, 'No Filter'),
                    ('surface_laplacian', BG_SL, 'Surface Laplacian'),
                ], start=nf_row):
                    ax = axes[row, col]
                    ax.set_facecolor(bg)

                    for grp, color, glbl in [('ADHD', ADHD_COLOR, 'ADHD'),
                                             ('TDC',  TDC_COLOR,  'TDC')]:
                        curves = []
                        for s in subjects:
                            if s['group'] != grp:
                                continue
                            sw = sweeps[tag].get(s['subject_id'])  # tag is 'no_filter' or 'surface_laplacian'
                            if sw and k in sw and metric in sw[k]:
                                curves.append(sw[k][metric])
                        if not curves:
                            continue
                        arr  = np.array(curves)
                        mean = arr.mean(axis=0)
                        n    = arr.shape[0]
                        ci   = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
                        ax.plot(densities, mean, color=color, linewidth=1.5, label=glbl)
                        ax.fill_between(densities, mean - ci, mean + ci,
                                        color=color, alpha=0.15)

                    ax.axvline(config.GRAPH_DENSITY, color='#888888',
                               linewidth=0.8, linestyle='--', alpha=0.7)
                    ax.set_xlim(densities[0], densities[-1])
                    ax.set_ylim(*ylim)
                    ax.tick_params(labelsize=5)
                    ax.grid(True, linewidth=0.3, alpha=0.4)

                    if row == 0:
                        ax.set_title(BAND_LABELS[b], fontsize=10, fontweight='bold')
                    if col == 0:
                        ax.set_ylabel(f'{GRAPH_LABELS[metric]}\n[{lbl}]', fontsize=6)
                    if row == nf_row and col == 0:
                        ax.legend(fontsize=5, loc='best', framealpha=0.8)

        fig.suptitle(
            f'Density Sweep — {MEASURE_LABELS[m]}  |  Raw (grey) vs Surface Laplacian (blue)\n'
            f'ADHD (n={n_adhd}) vs TDC (n={n_tdc})  |  '
            f'Mean ± 95% CI  |  Dashed = analysis threshold ({config.GRAPH_DENSITY})',
            fontsize=11,
        )
        fname = f'density_sweep_{m}_combined.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — Raw vs CSD pipeline comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_pipeline_comparison(subjects, out_dir):
    """
    Direct comparison of no-filter vs surface laplacian pipelines.

    Figure 1 — Betti curve overlay (B0 and B1):
        Each panel shows 4 lines: ADHD-nf, TDC-nf, ADHD-sl, TDC-sl.
        Makes it immediately obvious where surface laplacian changes the picture.

    Figure 2 — AUC difference heatmap:
        (ADHD_sl - ADHD_nf) and (TDC_sl - TDC_nf) as heatmaps
        across measure × band. Shows which combinations are most affected
        by the spatial filter.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Only use subjects that have both pipelines
    both = [s for s in subjects if s['features'] is not None
            and s['features_sl'] is not None]
    if len(both) < 4:
        print("  Skipping pipeline comparison — not enough subjects with both pipelines.")
        return

    n_adhd = sum(1 for s in both if s['group'] == 'ADHD')
    n_tdc  = sum(1 for s in both if s['group'] == 'TDC')
    eps    = np.linspace(0, 1.0, config.N_FILT_STEPS)

    RAW_ADHD_COLOR = '#e74c3c'   # solid red
    RAW_TDC_COLOR  = '#3498db'   # solid blue
    CSD_ADHD_COLOR = '#c0392b'   # darker red  (dashed)
    CSD_TDC_COLOR  = '#1a5276'   # darker blue (dashed)

    # ── Figure 1: Betti curve overlay ────────────────────────────────────────
    for homer in ['0', '1']:
        fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharey=False)
        fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.86, bottom=0.06,
                            left=0.06, right=0.98)

        # Global ymax and B1 x-zoom across both pipelines
        global_ymax = 0
        all_means   = []
        for pipeline in ['no_filter', 'surface_laplacian']:
            for m in MEASURES:
                for b in BANDS:
                    for grp in ['ADHD', 'TDC']:
                        c = _get_curve(both, grp, m, b, homer, pipeline)
                        if c.shape[0] > 0:
                            mean = c.mean(axis=0)
                            global_ymax = max(global_ymax, float(mean.max()))
                            all_means.append(mean)
        if global_ymax == 0:
            global_ymax = 1.0

        if homer == '1' and all_means:
            threshold  = 0.03 * global_ymax
            active     = np.zeros(config.N_FILT_STEPS, dtype=bool)
            for mean in all_means:
                active |= (mean > threshold)
            active_idx = np.where(active)[0]
            if len(active_idx) > 0:
                pad_steps = max(3, int(config.N_FILT_STEPS * 0.05))
                x_lo = eps[max(0, active_idx[0]  - pad_steps)]
                x_hi = eps[min(config.N_FILT_STEPS - 1, active_idx[-1] + pad_steps)]
            else:
                x_lo, x_hi = 0.0, 1.0
        else:
            x_lo, x_hi = 0.0, 1.0

        for row, m in enumerate(MEASURES):
            for col, b in enumerate(BANDS):
                ax = axes[row, col]

                styles = [
                    ('no_filter',        'ADHD', RAW_ADHD_COLOR, '-',  'ADHD no filter'),
                    ('no_filter',        'TDC',  RAW_TDC_COLOR,  '-',  'TDC no filter'),
                    ('surface_laplacian','ADHD', CSD_ADHD_COLOR, '--', 'ADHD sl'),
                    ('surface_laplacian','TDC',  CSD_TDC_COLOR,  '--', 'TDC sl'),
                ]
                for pipeline, grp, color, ls, lbl in styles:
                    curves = _get_curve(both, grp, m, b, homer, pipeline)
                    n = curves.shape[0]
                    if n == 0:
                        continue
                    mean = curves.mean(axis=0)
                    ax.plot(eps, mean, color=color, linewidth=1.6,
                            linestyle=ls, label=lbl)

                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(0, global_ymax * 1.08)
                ax.set_xlabel('ε', fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, linewidth=0.3, alpha=0.5)

                if row == 0:
                    ax.set_title(BAND_LABELS[b], fontsize=11, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(MEASURE_LABELS[m], fontsize=8)

        axes[0, 0].legend(fontsize=6, loc='upper right', framealpha=0.8)
        fig.suptitle(
            f'Betti-{homer} Curves — No Filter (solid) vs Surface Laplacian (dashed)\n'
            f'n={len(both)} subjects with both pipelines  '
            f'(ADHD={n_adhd}, TDC={n_tdc})',
            fontsize=12,
        )
        fname = f'pipeline_comparison_b{homer}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── Figure 2: AUC difference heatmap ─────────────────────────────────────
    for homer in ['0', '1']:
        field = f'auc_b{homer}'

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.88, bottom=0.08,
                            left=0.08, right=0.95)

        panels = [
            ('ADHD', 'SL - No Filter (ADHD)', axes[0, 0]),
            ('TDC',  'SL - No Filter (TDC)',  axes[0, 1]),
        ]

        all_diffs = []
        heatmaps  = {}
        for grp, title, ax in panels:
            matrix = np.zeros((len(MEASURES), len(BANDS)))
            for ri, m in enumerate(MEASURES):
                for ci, b in enumerate(BANDS):
                    nf_vals = _get_scalar(both, grp, m, b, field, 'no_filter')
                    sl_vals = _get_scalar(both, grp, m, b, field, 'surface_laplacian')
                    if nf_vals and sl_vals:
                        matrix[ri, ci] = np.mean(sl_vals) - np.mean(nf_vals)
            heatmaps[grp] = matrix
            all_diffs.extend(matrix.flatten())

        vmax = max(abs(v) for v in all_diffs) if all_diffs else 1.0
        vmin = -vmax

        for grp, title, ax in panels:
            matrix = heatmaps[grp]
            im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(BANDS)))
            ax.set_xticklabels([BAND_LABELS[b] for b in BANDS], fontsize=10)
            ax.set_yticks(range(len(MEASURES)))
            ax.set_yticklabels([MEASURE_LABELS[m] for m in MEASURES], fontsize=9)
            ax.set_title(title, fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, label='AUC difference')
            # Annotate cells
            for ri in range(len(MEASURES)):
                for ci in range(len(BANDS)):
                    ax.text(ci, ri, f'{matrix[ri, ci]:.2f}',
                            ha='center', va='center', fontsize=7,
                            color='white' if abs(matrix[ri, ci]) > vmax * 0.6 else 'black')

        # Difference of differences: (ADHD_sl - ADHD_nf) - (TDC_sl - TDC_nf)
        diff_of_diff = heatmaps['ADHD'] - heatmaps['TDC']
        vmax2 = max(abs(diff_of_diff.flatten())) if diff_of_diff.size else 1.0

        ax = axes[1, 0]
        im = ax.imshow(diff_of_diff, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax2, vmax=vmax2)
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels([BAND_LABELS[b] for b in BANDS], fontsize=10)
        ax.set_yticks(range(len(MEASURES)))
        ax.set_yticklabels([MEASURE_LABELS[m] for m in MEASURES], fontsize=9)
        ax.set_title('Interaction: (ADHD - TDC) change due to CSD', fontsize=10,
                     fontweight='bold')
        plt.colorbar(im, ax=ax, label='AUC diff-of-diff')
        for ri in range(len(MEASURES)):
            for ci in range(len(BANDS)):
                ax.text(ci, ri, f'{diff_of_diff[ri, ci]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(diff_of_diff[ri, ci]) > vmax2 * 0.6 else 'black')

        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5,
            'Blue = SL lowers AUC\nRed = SL raises AUC\n\n'
            'Interaction panel: where the\ngroup difference changes most\n'
            'after spatial filtering.',
            ha='center', va='center', fontsize=10,
            transform=axes[1, 1].transAxes,
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

        fig.suptitle(
            f'B{homer} AUC: Effect of Surface Laplacian (SL - No Filter)\n'
            f'n={len(both)} subjects with both pipelines',
            fontsize=12,
        )
        fname = f'pipeline_comparison_auc_b{homer}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — Topographic maps
# ══════════════════════════════════════════════════════════════════════════════

def _load_conn_matrices(subjects):
    """Load conn_matrices.npz from each subject's cache. Adds 'conn' key."""
    for s in subjects:
        sid        = s['subject_id']
        cache_path = os.path.join(config.RESULTS_ROOT, 'subjects', sid,
                                  '.cache', 'conn_matrices.npz')
        try:
            data = np.load(cache_path)
            conn = {tuple(k.split('__')): data[k] for k in data.files}
            s['conn'] = conn
        except Exception:
            s['conn'] = None
    return subjects


def _make_mne_info():
    """Create MNE Info object with standard 10-20 positions for TARGET_CHANNELS."""
    info = mne.create_info(
        ch_names=config.TARGET_CHANNELS,
        sfreq=config.SFREQ,
        ch_types='eeg',
    )
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')
    return info


def plot_topomaps_group(subjects, out_dir, pipeline='no_filter'):
    """
    Topographic maps of mean node strength per channel.
    Layout per figure: 3 rows (ADHD, TDC, ADHD-TDC) × 5 columns (bands).
    One figure per connectivity measure.
    Node strength = mean connectivity of each channel to all others.
    pipeline: 'no_filter' or 'surface_laplacian' (uses conn_matrices or conn_csd).
    """
    pipeline_label = 'No Filter' if pipeline == 'no_filter' else 'Surface Laplacian'
    fname_tag      = '' if pipeline == 'no_filter' else '_sl'

    os.makedirs(out_dir, exist_ok=True)

    # Load connectivity matrices
    subjects = _load_conn_matrices(subjects)

    # For Surface Laplacian, load csd conn matrices instead
    if pipeline == 'surface_laplacian':
        for s in subjects:
            sid        = s['subject_id']
            cache_path = os.path.join(config.RESULTS_ROOT, 'subjects', sid,
                                      '.cache', 'conn_csd.npz')
            try:
                data     = np.load(cache_path)
                s['conn'] = {tuple(k.split('__')): data[k] for k in data.files}
            except Exception:
                s['conn'] = None

    try:
        info = _make_mne_info()
    except Exception as e:
        print(f"  Skipping topomaps — could not create MNE info: {e}")
        return

    n_ch   = len(config.TARGET_CHANNELS)
    n_adhd = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc  = sum(1 for s in subjects if s['group'] == 'TDC')

    for m in MEASURES:
        fig, axes = plt.subplots(3, 5, figsize=(18, 11))
        fig.subplots_adjust(hspace=0.15, wspace=0.08, top=0.88, bottom=0.04,
                            left=0.06, right=0.96)

        # Collect all values for symmetric colormap
        all_adhd, all_tdc, all_diff = [], [], []

        strength_adhd = {}
        strength_tdc  = {}

        for col, b in enumerate(BANDS):
            key = (m, b)

            adhd_strengths, tdc_strengths = [], []
            for s in subjects:
                if not s.get('conn') or key not in s['conn']:
                    continue
                conn = s['conn'][key]
                # Node strength: mean connectivity to all other channels
                np.fill_diagonal(conn, 0)
                strength = conn.sum(axis=1) / (n_ch - 1)
                if s['group'] == 'ADHD':
                    adhd_strengths.append(strength)
                else:
                    tdc_strengths.append(strength)

            mean_adhd = np.mean(adhd_strengths, axis=0) if adhd_strengths else np.zeros(n_ch)
            mean_tdc  = np.mean(tdc_strengths,  axis=0) if tdc_strengths  else np.zeros(n_ch)
            diff      = mean_adhd - mean_tdc

            strength_adhd[b] = mean_adhd
            strength_tdc[b]  = mean_tdc
            all_adhd.extend(mean_adhd)
            all_tdc.extend(mean_tdc)
            all_diff.extend(diff)

        vmin_abs = min(min(all_adhd), min(all_tdc)) if all_adhd else 0
        vmax_abs = max(max(all_adhd), max(all_tdc)) if all_adhd else 1
        vmax_diff = max(abs(v) for v in all_diff) if all_diff else 1

        for col, b in enumerate(BANDS):
            mean_adhd = strength_adhd.get(b, np.zeros(n_ch))
            mean_tdc  = strength_tdc.get(b, np.zeros(n_ch))
            diff      = mean_adhd - mean_tdc

            row_configs = [
                (0, mean_adhd, 'RdYlBu_r', vmin_abs,   vmax_abs,   f'ADHD (n={n_adhd})'),
                (1, mean_tdc,  'RdYlBu_r', vmin_abs,   vmax_abs,   f'TDC  (n={n_tdc})'),
                (2, diff,      'RdBu_r',   -vmax_diff, vmax_diff,  'ADHD - TDC'),
            ]

            for row, vals, cmap, vmin, vmax, row_label in row_configs:
                ax = axes[row, col]
                try:
                    mne.viz.plot_topomap(
                        vals, info, axes=ax, show=False,
                        cmap=cmap, vlim=(vmin, vmax),
                        contours=4, sphere=0.1,
                        names=None,
                    )
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error:\n{e}', transform=ax.transAxes,
                            ha='center', va='center', fontsize=6)

                if col == 0:
                    ax.set_ylabel(row_label, fontsize=9)
                if row == 0:
                    ax.set_title(BAND_LABELS[b], fontsize=12, fontweight='bold')

        fig.suptitle(
            f'Node Strength Topomaps — {MEASURE_LABELS[m]}  |  {pipeline_label}\n'
            f'Row 1: ADHD mean   Row 2: TDC mean   '
            f'Row 3: Difference (red=ADHD higher, blue=TDC higher)',
            fontsize=11,
        )
        fname = f'topomaps_{m}{fname_tag}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


def _write_txt(out_dir, fname, lines):
    path = os.path.join(out_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — ICLabel artifact profile (ADHD vs TDC)
# ══════════════════════════════════════════════════════════════════════════════

_ICLABEL_CLASS_NAMES = [
    'brain', 'muscle artifact', 'eye blink', 'heart beat',
    'line noise', 'channel noise', 'other',
]
_ICLABEL_ARTIFACT_CLS = [
    'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise',
]


def _load_ica_reports(subjects):
    """Load ica_report.pkl from each subject's output dir. Adds 'ica_report' key."""
    for s in subjects:
        sid      = s['subject_id']
        pkl_path = os.path.join(config.RESULTS_ROOT, 'subjects', sid, 'ica_report.pkl')
        try:
            with open(pkl_path, 'rb') as f:
                s['ica_report'] = pickle.load(f)
        except Exception:
            s['ica_report'] = None
    return subjects


def plot_epoch_retention(subjects, out_dir):
    """
    Two panels showing AutoReject epoch retention per subject.

    Panel 1 — Scatter: epochs_before vs epochs_after coloured by group.
              Diagonal = 0 drop line.
    Panel 2 — Boxplot + jitter: epochs dropped (absolute count) by group.
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for s in subjects:
        log = _parse_run_log(s['subject_id'])
        if log and log.get('epochs_before') is not None:
            rows.append({
                'group':          s['group'],
                'epochs_before':  log['epochs_before'],
                'epochs_after':   log['epochs_after'],
                'epochs_dropped': log['epochs_dropped'],
            })

    if not rows:
        print("  Skipping epoch_retention — no log data found.")
        return

    adhd_rows = [r for r in rows if r['group'] == 'ADHD']
    tdc_rows  = [r for r in rows if r['group'] == 'TDC']
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('AutoReject Epoch Retention', fontsize=13, fontweight='bold', y=1.01)
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97, bottom=0.12, top=0.92)

    # ── Panel 1: before vs after scatter ─────────────────────────────────────
    ax = axes[0]
    all_vals = [r['epochs_before'] for r in rows] + [r['epochs_after'] for r in rows]
    lim = (0, max(all_vals) + 5)
    ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.4, label='No drop')

    for grp_rows, color, label in [
        (adhd_rows, ADHD_COLOR, f'ADHD (n={len(adhd_rows)})'),
        (tdc_rows,  TDC_COLOR,  f'TDC  (n={len(tdc_rows)})'),
    ]:
        bef = [r['epochs_before']  for r in grp_rows]
        aft = [r['epochs_after']   for r in grp_rows]
        ax.scatter(bef, aft, color=color, s=28, alpha=0.65, label=label)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Epochs before AutoReject', fontsize=10)
    ax.set_ylabel('Epochs after AutoReject',  fontsize=10)
    ax.set_title('Before vs After', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # ── Panel 2: epochs dropped boxplot ──────────────────────────────────────
    ax = axes[1]
    adhd_drop = [r['epochs_dropped'] for r in adhd_rows]
    tdc_drop  = [r['epochs_dropped'] for r in tdc_rows]

    for vals, x_pos, color, label in [
        (adhd_drop, 1, ADHD_COLOR, f'ADHD (n={len(adhd_rows)})'),
        (tdc_drop,  2, TDC_COLOR,  f'TDC  (n={len(tdc_rows)})'),
    ]:
        if not vals:
            continue
        bp = ax.boxplot([vals], positions=[x_pos], patch_artist=True,
                        widths=0.5, showfliers=False,
                        medianprops=dict(color='#333333', linewidth=1.5))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                   color=color, s=22, alpha=0.7, zorder=3, label=label)
        ax.text(x_pos, max(vals) + 0.3, f'μ={np.mean(vals):.1f}',
                ha='center', fontsize=8, color=color, fontweight='bold')

    ax.set_xlim(0.3, 2.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ADHD', 'TDC'], fontsize=10)
    ax.set_ylabel('Epochs dropped', fontsize=10)
    ax.set_title('Epochs Dropped\n(absolute count)', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=8)

    fig.savefig(os.path.join(out_dir, 'epoch_retention.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved epoch_retention.png")


def plot_bad_channel_map(subjects, out_dir):
    """
    Bar chart showing how often each of the 19 channels was flagged as
    bad by RANSAC, split by group (ADHD vs TDC).
    Reveals whether certain electrode sites are systematically problematic.
    """
    os.makedirs(out_dir, exist_ok=True)

    ch_count = {'ADHD': {ch: 0 for ch in config.TARGET_CHANNELS},
                'TDC':  {ch: 0 for ch in config.TARGET_CHANNELS}}
    n_subj   = {'ADHD': 0, 'TDC': 0}

    for s in subjects:
        grp = s['group']
        if grp not in n_subj:
            continue
        log = _parse_run_log(s['subject_id'])
        if not log:
            continue
        n_subj[grp] += 1
        for ch in log.get('bad_chs', []):
            if ch in ch_count[grp]:
                ch_count[grp][ch] += 1

    if n_subj['ADHD'] == 0 and n_subj['TDC'] == 0:
        print("  Skipping bad_channel_map — no log data.")
        return

    # Sort channels by anatomical region (frontal → central → parietal → occipital → temporal)
    region_order = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'O2',
    ]
    channels = [ch for ch in region_order if ch in config.TARGET_CHANNELS]

    # Normalise to % of subjects in each group
    adhd_pct = [100 * ch_count['ADHD'][ch] / max(n_subj['ADHD'], 1) for ch in channels]
    tdc_pct  = [100 * ch_count['TDC'][ch]  / max(n_subj['TDC'],  1) for ch in channels]

    x = np.arange(len(channels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, adhd_pct, w, color=ADHD_COLOR, alpha=0.75,
           label=f'ADHD (n={n_subj["ADHD"]})')
    ax.bar(x + w / 2, tdc_pct,  w, color=TDC_COLOR,  alpha=0.75,
           label=f'TDC  (n={n_subj["TDC"]})')

    # Region separators
    region_breaks = [7, 12, 17]   # after F8, T8, P8
    for brk in region_breaks:
        ax.axvline(brk - 0.5, color='grey', lw=0.7, ls='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(channels, fontsize=9)
    ax.set_ylabel('% of subjects flagged bad', fontsize=10)
    ax.set_title('RANSAC Bad Channel Frequency by Electrode', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    # Region labels
    region_labels = [('Frontal', 0, 6), ('Temporal/Central', 7, 11),
                     ('Parietal', 12, 16), ('Occipital', 17, 18)]
    for rname, start, end in region_labels:
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[1] * 1.02, rname,
                ha='center', fontsize=8, color='#555555', style='italic')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'bad_channel_map.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved bad_channel_map.png")


def plot_recording_length(subjects, out_dir):
    """
    A1 — Recording length distribution by group.

    epochs_before × 2 s = total recording time before any cleaning.
    Tests whether TDC recordings are systematically shorter, which would
    explain RANSAC instability and elevated bad-channel rates in TDC.

    Three panels:
      Left  — Boxplot + jitter of recording duration (seconds)
      Mid   — Histogram overlay
      Right — Sorted subject strip (rank plot) to show the full spread
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for s in subjects:
        log = _parse_run_log(s['subject_id'])
        if log and log.get('epochs_before') is not None:
            rows.append({'group': s['group'],
                         'secs':  log['epochs_before'] * 2,
                         'epochs': log['epochs_before']})

    if not rows:
        print("  Skipping recording_length — no log data.")
        return

    adhd = [r['secs'] for r in rows if r['group'] == 'ADHD']
    tdc  = [r['secs'] for r in rows if r['group'] == 'TDC']
    rng  = np.random.default_rng(42)

    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(adhd, tdc, alternative='two-sided')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'Recording Length by Group\n'
        f'ADHD μ={np.mean(adhd):.0f}s  |  TDC μ={np.mean(tdc):.0f}s  |  '
        f'Δ={np.mean(adhd)-np.mean(tdc):.0f}s  |  Mann-Whitney p={pval:.3f}',
        fontsize=12, fontweight='bold', y=1.02,
    )
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97, bottom=0.12, top=0.90)

    # ── Panel 1: boxplot + jitter ──────────────────────────────────────────────
    ax = axes[0]
    for vals, x_pos, color, label in [
        (adhd, 1, ADHD_COLOR, f'ADHD (n={len(adhd)})'),
        (tdc,  2, TDC_COLOR,  f'TDC  (n={len(tdc)})'),
    ]:
        bp = ax.boxplot([vals], positions=[x_pos], patch_artist=True,
                        widths=0.5, showfliers=False,
                        medianprops=dict(color='#333333', linewidth=1.5))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                   color=color, s=22, alpha=0.7, zorder=3, label=label)
        ax.text(x_pos, max(vals) + 4, f'μ={np.mean(vals):.0f}s',
                ha='center', fontsize=9, color=color, fontweight='bold')

    ax.set_xlim(0.3, 2.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ADHD', 'TDC'], fontsize=11)
    ax.set_ylabel('Recording duration (seconds)', fontsize=10)
    ax.set_title('Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=8)

    # Add significance bracket
    y_bracket = max(max(adhd), max(tdc)) + 12
    ax.annotate('', xy=(2, y_bracket), xytext=(1, y_bracket),
                arrowprops=dict(arrowstyle='-', color='#333333', lw=1.2))
    sig_str = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
    ax.text(1.5, y_bracket + 2, f'p={pval:.3f} {sig_str}',
            ha='center', fontsize=9, color='#333333')

    # ── Panel 2: histogram overlay ─────────────────────────────────────────────
    ax = axes[1]
    all_vals = adhd + tdc
    bins = np.linspace(min(all_vals) - 5, max(all_vals) + 5, 22)
    ax.hist(adhd, bins=bins, color=ADHD_COLOR, alpha=0.55, label=f'ADHD (n={len(adhd)})')
    ax.hist(tdc,  bins=bins, color=TDC_COLOR,  alpha=0.55, label=f'TDC  (n={len(tdc)})')
    ax.axvline(np.mean(adhd), color=ADHD_COLOR, lw=1.8, ls='--', alpha=0.9,
               label=f'ADHD mean {np.mean(adhd):.0f}s')
    ax.axvline(np.mean(tdc),  color=TDC_COLOR,  lw=1.8, ls='--', alpha=0.9,
               label=f'TDC mean {np.mean(tdc):.0f}s')
    ax.axvline(np.median(adhd), color=ADHD_COLOR, lw=1.2, ls=':', alpha=0.7)
    ax.axvline(np.median(tdc),  color=TDC_COLOR,  lw=1.2, ls=':', alpha=0.7)
    ax.set_xlabel('Recording duration (seconds)', fontsize=10)
    ax.set_ylabel('Number of subjects', fontsize=10)
    ax.set_title('Histogram', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    # ── Panel 3: rank plot (sorted strip) ─────────────────────────────────────
    ax = axes[2]
    adhd_sorted = sorted(adhd)
    tdc_sorted  = sorted(tdc)
    ax.plot(range(len(adhd_sorted)), adhd_sorted, 'o-', color=ADHD_COLOR,
            ms=4, lw=1.2, alpha=0.8, label=f'ADHD (n={len(adhd)})')
    ax.plot(range(len(tdc_sorted)),  tdc_sorted,  'o-', color=TDC_COLOR,
            ms=4, lw=1.2, alpha=0.8, label=f'TDC  (n={len(tdc)})')
    ax.axhline(np.mean(adhd), color=ADHD_COLOR, lw=1.2, ls='--', alpha=0.6)
    ax.axhline(np.mean(tdc),  color=TDC_COLOR,  lw=1.2, ls='--', alpha=0.6)
    ax.set_xlabel('Subject rank (sorted by duration)', fontsize=10)
    ax.set_ylabel('Recording duration (seconds)', fontsize=10)
    ax.set_title('Rank Plot', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.savefig(os.path.join(out_dir, 'recording_length.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved recording_length.png  "
          f"(ADHD mean={np.mean(adhd):.0f}s, TDC mean={np.mean(tdc):.0f}s, p={pval:.3f})")


def plot_length_vs_bad_channels(subjects, out_dir):
    """
    A2 — Recording length vs RANSAC bad channel count, coloured by group.

    Tests whether shorter recordings drive higher bad-channel rates via
    RANSAC spatial-covariance instability, independent of true signal quality.

    Three panels:
      Left  — Scatter with per-group regression lines
      Mid   — Scatter of recording length vs bad-channel RATE (n_bad / n_channels)
      Right — Within-group correlation coefficients with 95% CI (bootstrap)
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for s in subjects:
        log = _parse_run_log(s['subject_id'])
        if log and log.get('epochs_before') is not None:
            rows.append({
                'group':   s['group'],
                'secs':    log['epochs_before'] * 2,
                'n_bad':   log['n_bad'],
                'bad_rate': log['n_bad'] / len(config.TARGET_CHANNELS),
            })

    if not rows:
        print("  Skipping length_vs_bad_channels — no log data.")
        return

    adhd = [r for r in rows if r['group'] == 'ADHD']
    tdc  = [r for r in rows if r['group'] == 'TDC']

    from scipy.stats import pearsonr, spearmanr
    from numpy.polynomial.polynomial import polyfit as nppolyfit

    def _corr_boot(x, y, n_boot=2000, rng=None):
        """Bootstrap 95% CI for Spearman r."""
        if rng is None:
            rng = np.random.default_rng(42)
        r_obs, p_obs = spearmanr(x, y)
        boot_r = []
        idx = np.arange(len(x))
        for _ in range(n_boot):
            s_ = rng.choice(idx, size=len(idx), replace=True)
            r_, _ = spearmanr(np.array(x)[s_], np.array(y)[s_])
            boot_r.append(r_)
        ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
        return r_obs, p_obs, ci_lo, ci_hi

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97, bottom=0.13, top=0.88)

    def _scatter_with_fit(ax, x_key, y_key, ylabel, title):
        for grp, color, label in [(adhd, ADHD_COLOR, 'ADHD'), (tdc, TDC_COLOR, 'TDC')]:
            xs = np.array([r[x_key] for r in grp])
            ys = np.array([r[y_key] for r in grp])
            r, p, ci_lo, ci_hi = _corr_boot(xs, ys, rng=rng)

            ax.scatter(xs, ys, color=color, s=28, alpha=0.65,
                       label=f'{label} (n={len(grp)})  ρ={r:.2f}, p={p:.3f}')

            # Regression line
            if len(xs) > 2:
                m, b = np.polyfit(xs, ys, 1)
                x_line = np.linspace(xs.min(), xs.max(), 100)
                ax.plot(x_line, m * x_line + b, color=color, lw=1.8, alpha=0.8, ls='--')

        ax.set_xlabel('Recording duration (seconds)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7.5, loc='upper right')
        ax.grid(True, linewidth=0.3, alpha=0.4)

    # ── Panel 1: raw bad channel count ────────────────────────────────────────
    _scatter_with_fit(axes[0], 'secs', 'n_bad',
                      'RANSAC bad channels (count)',
                      'Recording Length vs Bad Channels')

    # ── Panel 2: bad channel rate ──────────────────────────────────────────────
    _scatter_with_fit(axes[1], 'secs', 'bad_rate',
                      f'Bad channel rate (n_bad / {len(config.TARGET_CHANNELS)} channels)',
                      'Recording Length vs Bad Channel Rate')

    # ── Panel 3: bootstrap correlation bar chart ───────────────────────────────
    ax = axes[2]
    results = {}
    for grp, label, color in [(adhd, 'ADHD', ADHD_COLOR), (tdc, 'TDC', TDC_COLOR)]:
        xs = np.array([r['secs']  for r in grp])
        ys = np.array([r['n_bad'] for r in grp])
        r, p, ci_lo, ci_hi = _corr_boot(xs, ys, rng=rng)
        results[label] = (r, p, ci_lo, ci_hi, color)

    x_pos = [0, 1]
    for xi, (label, (r, p, ci_lo, ci_hi, color)) in enumerate(results.items()):
        ax.bar(xi, r, color=color, alpha=0.7, width=0.5)
        ax.errorbar(xi, r, yerr=[[r - ci_lo], [ci_hi - r]],
                    fmt='none', color='#333333', capsize=6, lw=2)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(xi, ci_hi + 0.02, f'p={p:.3f}\n{sig}',
                ha='center', fontsize=8, color='#333333')

    ax.axhline(0, color='#333333', lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['ADHD', 'TDC'], fontsize=11)
    ax.set_ylabel("Spearman ρ  (length vs n_bad)\nwith 95% bootstrap CI", fontsize=9)
    ax.set_title('Within-Group Correlation\n(length → bad channels)', fontsize=11,
                 fontweight='bold')
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.text(0.5, -0.52,
            'Negative = longer recording → fewer bad channels\n(supports RANSAC instability hypothesis)',
            ha='center', fontsize=7.5, color='#555555', style='italic',
            transform=ax.transAxes)

    fig.suptitle('A2 — Does Recording Length Drive RANSAC Bad-Channel Rate?',
                 fontsize=12, fontweight='bold', y=1.01)

    fig.savefig(os.path.join(out_dir, 'length_vs_bad_channels.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved length_vs_bad_channels.png")


def plot_length_vs_ica(subjects, out_dir):
    """
    A3 — Recording length vs ICA exclusion count, coloured by group.

    If shorter recordings drive ICA instability (poor unmixing), we expect a
    negative correlation: longer recording → fewer components excluded.
    Separating exclusion type (eye vs line noise) reveals whether the effect
    is behavioural (eye blinks — plausible in ADHD) or environmental
    (line noise — suggests recording condition difference).
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for s in subjects:
        log = _parse_run_log(s['subject_id'])
        rpt = None
        pkl = os.path.join(config.RESULTS_ROOT, 'subjects', s['subject_id'], 'ica_report.pkl')
        if os.path.exists(pkl):
            try:
                import pickle as _pkl
                with open(pkl, 'rb') as f:
                    rpt = _pkl.load(f)
            except Exception:
                pass

        if log and log.get('epochs_before') is not None and log.get('ica_n') is not None:
            n_eye  = rpt['excluded_labels'].count('eye blink')   if rpt else None
            n_line = rpt['excluded_labels'].count('line noise')  if rpt else None
            rows.append({
                'group':   s['group'],
                'secs':    log['epochs_before'] * 2,
                'ica_n':   log['ica_n'],
                'n_eye':   n_eye,
                'n_line':  n_line,
            })

    if not rows:
        print("  Skipping length_vs_ica — no log data.")
        return

    adhd = [r for r in rows if r['group'] == 'ADHD']
    tdc  = [r for r in rows if r['group'] == 'TDC']

    from scipy.stats import spearmanr

    def _corr_boot(x, y, n_boot=2000, seed=42):
        rng = np.random.default_rng(seed)
        x, y = np.array(x), np.array(y)
        r_obs, p_obs = spearmanr(x, y)
        idx = np.arange(len(x))
        boot_r = [spearmanr(x[rng.choice(idx, len(idx), replace=True)],
                             y[rng.choice(idx, len(idx), replace=True)])[0]
                  for _ in range(n_boot)]
        ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
        return r_obs, p_obs, ci_lo, ci_hi

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97, bottom=0.13, top=0.88)
    fig.suptitle('A3 — Does Recording Length Drive ICA Exclusion Count?',
                 fontsize=12, fontweight='bold', y=1.01)

    # ── Panel 1: total ICA exclusions ─────────────────────────────────────────
    ax = axes[0]
    for grp, color, label in [(adhd, ADHD_COLOR, 'ADHD'), (tdc, TDC_COLOR, 'TDC')]:
        xs = np.array([r['secs']  for r in grp])
        ys = np.array([r['ica_n'] for r in grp])
        r, p, _, _ = _corr_boot(xs, ys)
        ax.scatter(xs, ys, color=color, s=28, alpha=0.65,
                   label=f'{label} (n={len(grp)})  rho={r:.2f}, p={p:.3f}')
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_line, m * x_line + b, color=color, lw=1.8, ls='--', alpha=0.8)

    ax.set_xlabel('Recording duration (seconds)', fontsize=10)
    ax.set_ylabel('ICA components excluded (total)', fontsize=10)
    ax.set_title('Total ICA Exclusions', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # ── Panel 2: eye vs line noise breakdown ──────────────────────────────────
    ax = axes[1]
    has_rpt = [r for r in rows if r['n_eye'] is not None]
    if has_rpt:
        adhd_rpt = [r for r in has_rpt if r['group'] == 'ADHD']
        tdc_rpt  = [r for r in has_rpt if r['group'] == 'TDC']

        markers = {'eye blink': ('o', 'n_eye',  'Eye blink'),
                   'line noise': ('s', 'n_line', 'Line noise')}

        for grp, color, grp_label in [(adhd_rpt, ADHD_COLOR, 'ADHD'),
                                       (tdc_rpt,  TDC_COLOR,  'TDC')]:
            xs = np.array([r['secs'] for r in grp])
            for marker, key, type_label in [('o', 'n_eye', 'Eye'), ('s', 'n_line', 'Line noise')]:
                ys = np.array([r[key] for r in grp])
                ax.scatter(xs, ys, color=color, marker=marker, s=28, alpha=0.55,
                           label=f'{grp_label} {type_label}')
                m, b = np.polyfit(xs, ys, 1)
                x_line = np.linspace(xs.min(), xs.max(), 100)
                ax.plot(x_line, m * x_line + b, color=color, lw=1.2,
                        ls='--' if key == 'n_eye' else ':', alpha=0.7)

    ax.set_xlabel('Recording duration (seconds)', fontsize=10)
    ax.set_ylabel('Components excluded (by type)', fontsize=10)
    ax.set_title('Eye Blink vs Line Noise\n(o = eye, s = line noise)', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # ── Panel 3: bootstrap correlation bars — total, eye, line noise ──────────
    ax = axes[2]
    metrics = [('ica_n', 'Total'),   ('n_eye', 'Eye blink'), ('n_line', 'Line noise')]
    x_base  = np.array([0, 1.8, 3.6])
    width   = 0.7

    for gi, (grp, color, grp_label) in enumerate([(adhd, ADHD_COLOR, 'ADHD'),
                                                   (tdc,  TDC_COLOR,  'TDC')]):
        for mi, (key, mlabel) in enumerate(metrics):
            valid = [r for r in grp if r[key] is not None]
            if len(valid) < 5:
                continue
            xs = np.array([r['secs'] for r in valid])
            ys = np.array([r[key]    for r in valid])
            r, p, ci_lo, ci_hi = _corr_boot(xs, ys)
            xpos = x_base[mi] + gi * width * 0.55

            bar = ax.bar(xpos, r, width=width * 0.5, color=color, alpha=0.75,
                         label=grp_label if mi == 0 else None)
            ax.errorbar(xpos, r, yerr=[[r - ci_lo], [ci_hi - r]],
                        fmt='none', color='#333333', capsize=4, lw=1.5)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            if sig:
                ax.text(xpos, ci_hi + 0.02, sig, ha='center', fontsize=9, color='#333333')

    ax.axhline(0, color='#333333', lw=0.8)
    ax.set_xticks(x_base + width * 0.275)
    ax.set_xticklabels(['Total', 'Eye blink', 'Line noise'], fontsize=10)
    ax.set_ylabel('Spearman rho (length vs exclusions)\nwith 95% bootstrap CI', fontsize=9)
    ax.set_title('Within-Group Correlations\nby Exclusion Type', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.55, 0.55)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.text(0.5, -0.50,
            'Negative = longer recording -> fewer exclusions (instability hypothesis)',
            ha='center', fontsize=7.5, color='#555555', style='italic',
            transform=ax.transAxes)

    fig.savefig(os.path.join(out_dir, 'length_vs_ica.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved length_vs_ica.png")


def plot_length_vs_drop_rate(subjects, out_dir):
    """
    A4 — Recording length vs epoch drop RATE (proportion), coloured by group.

    Absolute drop count is confounded by recording length — a subject with
    160 epochs dropping 10 is much cleaner than one with 40 dropping 10.
    Proportion controls for this. If shorter recordings have higher drop rates,
    preprocessing is genuinely less stable with less data. If drop rate is
    independent of length, the epoch quality differences are real, not
    a length artefact.

    Also plots drop rate distribution by group to show whether TDC
    genuinely has noisier epochs.
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for s in subjects:
        log = _parse_run_log(s['subject_id'])
        if log and log.get('epochs_before') is not None and log['epochs_before'] > 0:
            rows.append({
                'group':     s['group'],
                'secs':      log['epochs_before'] * 2,
                'drop_rate': log['epochs_dropped'] / log['epochs_before'],
                'dropped':   log['epochs_dropped'],
                'before':    log['epochs_before'],
            })

    if not rows:
        print("  Skipping length_vs_drop_rate — no log data.")
        return

    adhd = [r for r in rows if r['group'] == 'ADHD']
    tdc  = [r for r in rows if r['group'] == 'TDC']

    from scipy.stats import spearmanr, mannwhitneyu

    def _corr_boot(x, y, n_boot=2000, seed=42):
        rng = np.random.default_rng(seed)
        x, y = np.array(x), np.array(y)
        r_obs, p_obs = spearmanr(x, y)
        idx = np.arange(len(x))
        boot_r = [spearmanr(x[rng.choice(idx, len(idx), replace=True)],
                             y[rng.choice(idx, len(idx), replace=True)])[0]
                  for _ in range(n_boot)]
        ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
        return r_obs, p_obs, ci_lo, ci_hi

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97, bottom=0.13, top=0.88)
    fig.suptitle('A4 — Does Recording Length Drive Epoch Drop Rate?',
                 fontsize=12, fontweight='bold', y=1.01)

    # ── Panel 1: scatter length vs drop rate ──────────────────────────────────
    ax = axes[0]
    for grp, color, label in [(adhd, ADHD_COLOR, 'ADHD'), (tdc, TDC_COLOR, 'TDC')]:
        xs = np.array([r['secs']      for r in grp])
        ys = np.array([r['drop_rate'] for r in grp]) * 100   # as %
        r, p, _, _ = _corr_boot(xs, ys / 100)
        ax.scatter(xs, ys, color=color, s=28, alpha=0.65,
                   label=f'{label} (n={len(grp)})  rho={r:.2f}, p={p:.3f}')
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_line, m * x_line + b, color=color, lw=1.8, ls='--', alpha=0.8)

    ax.set_xlabel('Recording duration (seconds)', fontsize=10)
    ax.set_ylabel('Epoch drop rate (%)', fontsize=10)
    ax.set_title('Recording Length vs Drop Rate', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # ── Panel 2: drop rate distribution by group ──────────────────────────────
    ax = axes[1]
    adhd_dr = np.array([r['drop_rate'] for r in adhd]) * 100
    tdc_dr  = np.array([r['drop_rate'] for r in tdc])  * 100
    stat, pval = mannwhitneyu(adhd_dr, tdc_dr, alternative='two-sided')

    bins = np.linspace(0, max(adhd_dr.max(), tdc_dr.max()) + 2, 22)
    ax.hist(adhd_dr, bins=bins, color=ADHD_COLOR, alpha=0.55,
            label=f'ADHD  mean={adhd_dr.mean():.1f}%')
    ax.hist(tdc_dr,  bins=bins, color=TDC_COLOR,  alpha=0.55,
            label=f'TDC   mean={tdc_dr.mean():.1f}%')
    ax.axvline(adhd_dr.mean(), color=ADHD_COLOR, lw=1.8, ls='--', alpha=0.9)
    ax.axvline(tdc_dr.mean(),  color=TDC_COLOR,  lw=1.8, ls='--', alpha=0.9)
    ax.set_xlabel('Epoch drop rate (%)', fontsize=10)
    ax.set_ylabel('Number of subjects', fontsize=10)
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
    ax.set_title(f'Drop Rate Distribution by Group\nMann-Whitney p={pval:.3f} {sig}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    # ── Panel 3: boxplot + jitter of drop rate ────────────────────────────────
    ax = axes[2]
    for vals, x_pos, color, label in [
        (adhd_dr, 1, ADHD_COLOR, f'ADHD (n={len(adhd)})'),
        (tdc_dr,  2, TDC_COLOR,  f'TDC  (n={len(tdc)})'),
    ]:
        bp = ax.boxplot([vals], positions=[x_pos], patch_artist=True,
                        widths=0.5, showfliers=False,
                        medianprops=dict(color='#333333', linewidth=1.5))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                   color=color, s=22, alpha=0.7, zorder=3, label=label)
        ax.text(x_pos, vals.max() + 0.5,
                f'mean={vals.mean():.1f}%\nmed={np.median(vals):.1f}%',
                ha='center', fontsize=8, color=color, fontweight='bold')

    ax.set_xlim(0.3, 2.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ADHD', 'TDC'], fontsize=11)
    ax.set_ylabel('Epoch drop rate (%)', fontsize=10)
    ax.set_title(f'Drop Rate by Group\np={pval:.3f} {sig}', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=8)

    fig.savefig(os.path.join(out_dir, 'length_vs_drop_rate.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved length_vs_drop_rate.png  "
          f"(ADHD mean={adhd_dr.mean():.1f}%, TDC mean={tdc_dr.mean():.1f}%, p={pval:.3f})")


def plot_edge_weight_variance(subjects, out_dir):
    """
    B1 — Edge weight variance per subject, for both pipelines.

    Three figures are saved:
      edge_weight_variance_global.png  — mean variance across all edges, NF vs SL,
                                         boxplot by group + bad-channel confound check
      edge_weight_variance_electrode.png — per-electrode row variance (each electrode's
                                           variance across its 18 connections), group
                                           mean bar chart for NF and SL side by side
      edge_weight_variance_pipeline.png  — NF vs SL variance ratio per subject showing
                                           how much SL sharpens spatial heterogeneity

    High variance = spatially heterogeneous network (some edges strong, some weak).
    Low variance = uniform, spatially predictable signal — what RANSAC expects.
    SL suppresses volume conduction so local differences become more visible.
    """
    os.makedirs(out_dir, exist_ok=True)

    from scipy.stats import mannwhitneyu, spearmanr
    from matplotlib.patches import Patch

    # ── Load connectivity matrices for both pipelines ─────────────────────────
    rows = []
    for s in subjects:
        sid       = s['subject_id']
        cache_dir = os.path.join(config.RESULTS_ROOT, 'subjects', sid, '.cache')
        log       = _parse_run_log(sid)

        entry = {'subject_id': sid, 'group': s['group'],
                 'n_bad': log['n_bad'] if log else None}

        for pipe_key, fname in [('nf', 'conn_matrices.npz'), ('sl', 'conn_csd.npz')]:
            npz_path = os.path.join(cache_dir, fname)
            if not os.path.exists(npz_path):
                entry[pipe_key] = None
                continue
            try:
                d = np.load(npz_path)
            except Exception:
                entry[pipe_key] = None
                continue

            # global variance (all edges pooled) and per-electrode row variance
            all_vars   = []
            elec_vars  = {ch: [] for ch in config.TARGET_CHANNELS}  # variance of each electrode's row
            n_ch       = len(config.TARGET_CHANNELS)

            for raw_key in d.files:
                mat = d[raw_key]
                if mat.ndim != 2 or mat.shape[0] != n_ch:
                    continue
                # global: upper triangle
                tri = mat[np.triu_indices(n_ch, k=1)]
                tri = tri[np.isfinite(tri)]
                if len(tri):
                    all_vars.append(float(np.var(tri)))
                # per-electrode: variance of each row (excluding diagonal)
                for ei, ch in enumerate(config.TARGET_CHANNELS):
                    row = np.array([mat[ei, j] for j in range(n_ch) if j != ei])
                    row = row[np.isfinite(row)]
                    if len(row):
                        elec_vars[ch].append(float(np.var(row)))

            if not all_vars:
                entry[pipe_key] = None
                continue

            entry[pipe_key] = {
                'mean_var':  float(np.mean(all_vars)),
                'elec_vars': {ch: float(np.mean(v)) if v else np.nan
                              for ch, v in elec_vars.items()},
            }

        if entry.get('nf') is not None or entry.get('sl') is not None:
            rows.append(entry)

    if not rows:
        print("  Skipping edge_weight_variance — no connectivity files found.")
        return

    rng = np.random.default_rng(42)

    def _grp(pipe_key, group):
        return [r for r in rows if r['group'] == group and r.get(pipe_key) is not None]

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Global variance: NF vs SL, by group + bad-channel check
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97, bottom=0.13, top=0.90)
    fig.suptitle('B1 — Global Edge Weight Variance by Group & Pipeline',
                 fontsize=12, fontweight='bold', y=1.01)

    for col, (pipe_key, pipe_label) in enumerate([('nf', 'No Filter'), ('sl', 'Surface Laplacian')]):
        ax = axes[col]
        grp_data = {}
        for grp, color in [('ADHD', ADHD_COLOR), ('TDC', TDC_COLOR)]:
            r_list = _grp(pipe_key, grp)
            vals   = [r[pipe_key]['mean_var'] for r in r_list]
            grp_data[grp] = (vals, color, len(r_list))

        x_pos = {'ADHD': 1, 'TDC': 2}
        for grp, (vals, color, n) in grp_data.items():
            if not vals:
                continue
            bp = ax.boxplot([vals], positions=[x_pos[grp]], patch_artist=True,
                            widths=0.5, showfliers=False,
                            medianprops=dict(color='#333333', linewidth=1.5))
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.5)
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), x_pos[grp]) + jitter, vals,
                       color=color, s=22, alpha=0.7, zorder=3,
                       label=f'{grp} (n={n})')
            ax.text(x_pos[grp], max(vals) * 1.02,
                    f'mean={np.mean(vals):.4f}',
                    ha='center', fontsize=8, color=color, fontweight='bold')

        adhd_v = grp_data['ADHD'][0]
        tdc_v  = grp_data['TDC'][0]
        if adhd_v and tdc_v:
            _, pval = mannwhitneyu(adhd_v, tdc_v, alternative='two-sided')
            sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
            y_br = max(max(adhd_v), max(tdc_v)) * 1.06
            ax.annotate('', xy=(2, y_br), xytext=(1, y_br),
                        arrowprops=dict(arrowstyle='-', color='#333333', lw=1.2))
            ax.text(1.5, y_br * 1.01, f'p={pval:.3f} {sig}', ha='center', fontsize=9)

        ax.set_xlim(0.3, 2.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['ADHD', 'TDC'], fontsize=11)
        ax.set_ylabel('Mean edge weight variance', fontsize=9)
        ax.set_title(f'{pipe_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    # Panel 3: bad-channel confound check (NF only)
    ax = axes[2]
    for grp, color in [('ADHD', ADHD_COLOR), ('TDC', TDC_COLOR)]:
        valid = [r for r in _grp('nf', grp) if r['n_bad'] is not None]
        if not valid:
            continue
        xs = np.array([r['n_bad'] for r in valid], dtype=float)
        ys = np.array([r['nf']['mean_var'] for r in valid])
        r_s, p_s = spearmanr(xs, ys)
        jx = rng.uniform(-0.08, 0.08, len(xs))
        ax.scatter(xs + jx, ys, color=color, s=28, alpha=0.65,
                   label=f'{grp}  rho={r_s:.2f}, p={p_s:.3f}')
        if len(set(xs)) > 2:
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(x_line, m * x_line + b, color=color, lw=1.8, ls='--', alpha=0.8)

    ax.set_xlabel('RANSAC bad channels', fontsize=10)
    ax.set_ylabel('Mean edge weight variance (No Filter)', fontsize=9)
    ax.set_title('Variance vs Bad Channels\n(interpolation homogenisation check)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.text(0.5, -0.14, 'Negative rho = more interpolation -> lower variance',
            ha='center', fontsize=7.5, color='#555555', style='italic',
            transform=ax.transAxes)

    fig.savefig(os.path.join(out_dir, 'edge_weight_variance_global.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved edge_weight_variance_global.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Per-electrode variance: group mean bar chart, NF + SL
    # ══════════════════════════════════════════════════════════════════════════
    region_order = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'O2',
    ]
    channels = [ch for ch in region_order if ch in config.TARGET_CHANNELS]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.55, left=0.07, right=0.97, bottom=0.08, top=0.92)
    fig.suptitle('B1 — Per-Electrode Edge Weight Variance\n'
                 '(each electrode\'s variance across its 18 connections, avg over 15 measure×band matrices)',
                 fontsize=12, fontweight='bold')

    region_breaks = [7, 12, 17]

    for row_idx, (pipe_key, pipe_label) in enumerate([('nf', 'No Filter'),
                                                       ('sl', 'Surface Laplacian')]):
        ax = axes[row_idx]
        x  = np.arange(len(channels))
        w  = 0.38

        for grp, color, offset in [('ADHD', ADHD_COLOR, -w/2), ('TDC', TDC_COLOR, w/2)]:
            r_list = _grp(pipe_key, grp)
            if not r_list:
                continue
            means = []
            sems  = []
            for ch in channels:
                ch_vals = [r[pipe_key]['elec_vars'][ch] for r in r_list
                           if np.isfinite(r[pipe_key]['elec_vars'].get(ch, np.nan))]
                means.append(np.mean(ch_vals) if ch_vals else 0)
                sems.append(np.std(ch_vals) / np.sqrt(len(ch_vals)) if len(ch_vals) > 1 else 0)

            ax.bar(x + offset, means, width=w, color=color, alpha=0.75,
                   label=f'{grp} (n={len(r_list)})')
            ax.errorbar(x + offset, means, yerr=sems,
                        fmt='none', color='#333333', capsize=2, lw=1.0)

        for brk in region_breaks:
            ax.axvline(brk - 0.5, color='grey', lw=0.7, ls='--', alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(channels, fontsize=8.5)
        ax.set_ylabel('Mean row variance', fontsize=9)
        ax.set_title(f'{pipe_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

        region_labels = [('Frontal', 0, 6), ('Temporal/Central', 7, 11),
                         ('Parietal', 12, 16), ('Occipital', 17, 18)]
        for rname, start, end in region_labels:
            ax.text((start + end) / 2, ax.get_ylim()[1] * 1.03, rname,
                    ha='center', fontsize=8, color='#555555', style='italic')

    fig.savefig(os.path.join(out_dir, 'edge_weight_variance_electrode.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved edge_weight_variance_electrode.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — NF vs SL variance ratio per subject (SL sharpening effect)
    # ══════════════════════════════════════════════════════════════════════════
    both = [r for r in rows if r.get('nf') is not None and r.get('sl') is not None]
    if both:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.subplots_adjust(wspace=0.38, left=0.08, right=0.97, bottom=0.12, top=0.90)
        fig.suptitle('B1 — Surface Laplacian Sharpening Effect on Edge Variance\n'
                     '(ratio = SL variance / NF variance per subject)',
                     fontsize=12, fontweight='bold', y=1.01)

        adhd_b = [r for r in both if r['group'] == 'ADHD']
        tdc_b  = [r for r in both if r['group'] == 'TDC']

        # Panel 1: ratio boxplot
        ax = axes[0]
        for grp_rows, x_pos, color, label in [
            (adhd_b, 1, ADHD_COLOR, f'ADHD (n={len(adhd_b)})'),
            (tdc_b,  2, TDC_COLOR,  f'TDC  (n={len(tdc_b)})'),
        ]:
            ratios = [r['sl']['mean_var'] / r['nf']['mean_var']
                      for r in grp_rows if r['nf']['mean_var'] > 0]
            if not ratios:
                continue
            bp = ax.boxplot([ratios], positions=[x_pos], patch_artist=True,
                            widths=0.5, showfliers=False,
                            medianprops=dict(color='#333333', linewidth=1.5))
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.5)
            jitter = rng.uniform(-0.15, 0.15, len(ratios))
            ax.scatter(np.full(len(ratios), x_pos) + jitter, ratios,
                       color=color, s=22, alpha=0.7, zorder=3, label=label)
            ax.text(x_pos, max(ratios) * 1.02,
                    f'mean={np.mean(ratios):.2f}x',
                    ha='center', fontsize=9, color=color, fontweight='bold')

        ax.axhline(1.0, color='#333333', lw=1.0, ls='--', alpha=0.5,
                   label='ratio=1 (no change)')
        ax.set_xlim(0.3, 2.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['ADHD', 'TDC'], fontsize=11)
        ax.set_ylabel('SL variance / NF variance', fontsize=10)
        ax.set_title('SL Sharpening Ratio\n(>1 = SL increases heterogeneity)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

        # Panel 2: NF vs SL scatter coloured by group
        ax = axes[1]
        for grp_rows, color, label in [(adhd_b, ADHD_COLOR, 'ADHD'),
                                        (tdc_b,  TDC_COLOR,  'TDC')]:
            xs = [r['nf']['mean_var'] for r in grp_rows]
            ys = [r['sl']['mean_var'] for r in grp_rows]
            ax.scatter(xs, ys, color=color, s=28, alpha=0.65, label=label)

        lim_max = max(max(r['nf']['mean_var'] for r in both),
                      max(r['sl']['mean_var'] for r in both)) * 1.05
        ax.plot([0, lim_max], [0, lim_max], 'k--', lw=0.8, alpha=0.4,
                label='NF=SL line')
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_xlabel('No Filter edge variance', fontsize=10)
        ax.set_ylabel('Surface Laplacian edge variance', fontsize=10)
        ax.set_title('NF vs SL Variance per Subject', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.4)

        fig.savefig(os.path.join(out_dir, 'edge_weight_variance_pipeline.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved edge_weight_variance_pipeline.png")


def plot_interpolation_connectivity_bias(subjects, out_dir):
    """
    B2 — Connectivity of interpolated channels vs same channel in non-interpolated subjects.

    When RANSAC flags a channel as bad, MNE replaces it with a weighted spatial
    average of neighbouring channels. This makes the interpolated channel's signal
    mathematically derived from its neighbours, artificially inflating coherence
    and correlation with those neighbours in the final connectivity matrix.

    This function tests whether interpolated channels show higher mean connectivity
    than the same channel in subjects where it was NOT interpolated, and whether
    this inflation falls disproportionately on ADHD vs TDC given their different
    bad-channel patterns.

    Figure 1 — Per-channel: connectivity inflation (interpolated - non-interpolated mean)
    Figure 2 — Group breakdown: which channels inflated in ADHD vs TDC
    Figure 3 — Subject-level: total inflation load per subject by group
    """
    os.makedirs(out_dir, exist_ok=True)

    from scipy.stats import mannwhitneyu
    from collections import defaultdict

    n_ch = len(config.TARGET_CHANNELS)

    # ── Load per-subject data ─────────────────────────────────────────────────
    ch_records = defaultdict(list)   # ch -> [{mean_conn, interpolated, group, sid}]
    subj_load  = []                  # per-subject inflation load

    for s in subjects:
        sid = s['subject_id']
        log = pg_parse = _parse_run_log(sid)
        if not log:
            continue
        bad_chs = set(log.get('bad_chs', []))

        npz = os.path.join(config.RESULTS_ROOT, 'subjects', sid, '.cache', 'conn_matrices.npz')
        if not os.path.exists(npz):
            continue
        try:
            d = np.load(npz)
        except Exception:
            continue

        mats = [d[k] for k in d.files if d[k].ndim == 2 and d[k].shape == (n_ch, n_ch)]
        if not mats:
            continue
        avg_mat = np.nanmean(mats, axis=0)

        ch_conn = {}
        for ei, ch in enumerate(config.TARGET_CHANNELS):
            row = np.array([avg_mat[ei, j] for j in range(n_ch) if j != ei])
            row = row[np.isfinite(row)]
            if len(row):
                ch_conn[ch] = float(np.mean(row))
                ch_records[ch].append({
                    'mean_conn':    ch_conn[ch],
                    'interpolated': ch in bad_chs,
                    'group':        s['group'],
                    'sid':          sid,
                })

        subj_load.append({
            'sid':   sid,
            'group': s['group'],
            'n_bad': log['n_bad'],
            'bad_chs': bad_chs,
            'ch_conn': ch_conn,
        })

    if not ch_records:
        print("  Skipping interpolation_connectivity_bias — no data.")
        return

    # ── Compute per-channel inflation stats ───────────────────────────────────
    region_order = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
    ]
    channels = [ch for ch in region_order if ch in config.TARGET_CHANNELS]

    inflation = {}   # ch -> {diff, interp_mean, normal_mean, n_interp}
    for ch in channels:
        recs   = ch_records[ch]
        interp = [r['mean_conn'] for r in recs if r['interpolated']]
        normal = [r['mean_conn'] for r in recs if not r['interpolated']]
        if len(interp) < 2 or not normal:
            inflation[ch] = None
            continue
        diff = np.mean(interp) - np.mean(normal)
        _, pval = mannwhitneyu(interp, normal, alternative='greater')
        inflation[ch] = {
            'diff':         diff,
            'interp_mean':  np.mean(interp),
            'normal_mean':  np.mean(normal),
            'n_interp':     len(interp),
            'pval':         pval,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Per-channel inflation bar chart
    # ══════════════════════════════════════════════════════════════════════════
    valid_chs = [ch for ch in channels if inflation.get(ch) is not None]
    diffs     = [inflation[ch]['diff']     for ch in valid_chs]
    n_interps = [inflation[ch]['n_interp'] for ch in valid_chs]
    pvals     = [inflation[ch]['pval']     for ch in valid_chs]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.55, left=0.08, right=0.97, bottom=0.09, top=0.92)
    fig.suptitle('B2 — Connectivity Inflation in Interpolated Channels\n'
                 '(mean connectivity of interpolated channel minus same channel in non-interpolated subjects)',
                 fontsize=12, fontweight='bold')

    region_breaks = [7, 12, 17]

    # Top: inflation magnitude
    ax = axes[0]
    x  = np.arange(len(valid_chs))
    colors_bar = ['#c0392b' if d > 0 else '#2980b9' for d in diffs]
    bars = ax.bar(x, diffs, color=colors_bar, alpha=0.75, width=0.65)

    for xi, (ch, d, p, ni) in enumerate(zip(valid_chs, diffs, pvals, n_interps)):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        label = f'n={ni}'
        if sig:
            label += f'\n{sig}'
        yoff = d + 0.002 if d >= 0 else d - 0.008
        ax.text(xi, yoff, label, ha='center', fontsize=7, color='#333333')

    for brk in region_breaks:
        ax.axvline(brk - 0.5, color='grey', lw=0.7, ls='--', alpha=0.5)
    ax.axhline(0, color='#333333', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_chs, fontsize=9)
    ax.set_ylabel('Connectivity inflation\n(interpolated − normal)', fontsize=9)
    ax.set_title('Mean Connectivity Inflation per Channel  (red = inflated, n = subjects interpolated)',
                 fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    region_labels = [('Frontal', 0, 6), ('Temporal/Central', 7, 11),
                     ('Parietal', 12, 16), ('Occipital', 17, 18)]
    for rname, start, end in region_labels:
        valid_in_region = [i for i, ch in enumerate(valid_chs)
                           if start <= channels.index(ch) <= end]
        if valid_in_region:
            mid = np.mean(valid_in_region)
            ax.text(mid, ax.get_ylim()[1] * 1.04, rname,
                    ha='center', fontsize=8, color='#555555', style='italic')

    # Bottom: absolute connectivity — interpolated vs normal side by side
    ax = axes[1]
    w = 0.35
    interp_means = [inflation[ch]['interp_mean'] for ch in valid_chs]
    normal_means = [inflation[ch]['normal_mean'] for ch in valid_chs]
    ax.bar(x - w/2, normal_means,  w, color='#7f8c8d', alpha=0.75, label='Not interpolated')
    ax.bar(x + w/2, interp_means, w, color='#c0392b',  alpha=0.75, label='Interpolated')

    for brk in region_breaks:
        ax.axvline(brk - 0.5, color='grey', lw=0.7, ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_chs, fontsize=9)
    ax.set_ylabel('Mean connectivity strength', fontsize=9)
    ax.set_title('Absolute Connectivity: Interpolated vs Not Interpolated',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    fig.savefig(os.path.join(out_dir, 'interpolation_bias_channels.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved interpolation_bias_channels.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Group breakdown: inflation split by ADHD vs TDC
    # ══════════════════════════════════════════════════════════════════════════
    # Focus on channels with >=3 interpolated subjects in either group
    group_inflation = {}
    for ch in channels:
        recs = ch_records[ch]
        by_grp = {}
        for grp in ['ADHD', 'TDC']:
            interp = [r['mean_conn'] for r in recs
                      if r['interpolated'] and r['group'] == grp]
            normal = [r['mean_conn'] for r in recs
                      if not r['interpolated'] and r['group'] == grp]
            if len(interp) >= 2 and normal:
                by_grp[grp] = {
                    'diff':     np.mean(interp) - np.mean(normal),
                    'n_interp': len(interp),
                }
        if by_grp:
            group_inflation[ch] = by_grp

    focus_chs = [ch for ch in channels if ch in group_inflation]

    if focus_chs:
        fig, ax = plt.subplots(figsize=(15, 5))
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.88)

        x   = np.arange(len(focus_chs))
        w   = 0.38
        for grp, color, offset in [('ADHD', ADHD_COLOR, -w/2), ('TDC', TDC_COLOR, w/2)]:
            diffs_g  = []
            n_interp = []
            for ch in focus_chs:
                gi = group_inflation[ch].get(grp)
                diffs_g.append(gi['diff']     if gi else 0)
                n_interp.append(gi['n_interp'] if gi else 0)

            bars = ax.bar(x + offset, diffs_g, w, color=color, alpha=0.75,
                          label=grp)
            for xi, (d, ni) in enumerate(zip(diffs_g, n_interp)):
                if ni > 0:
                    ax.text(xi + offset, d + (0.002 if d >= 0 else -0.007),
                            f'n={ni}', ha='center', fontsize=7, color='#333333')

        for brk in region_breaks:
            ax.axvline(brk - 0.5, color='grey', lw=0.7, ls='--', alpha=0.5)
        ax.axhline(0, color='#333333', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(focus_chs, fontsize=9)
        ax.set_ylabel('Connectivity inflation (interpolated − normal)', fontsize=10)
        ax.set_title('B2 — Interpolation Connectivity Inflation by Group\n'
                     '(which group gets artificially inflated connectivity at each site)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

        for rname, start, end in region_labels:
            valid_in = [i for i, ch in enumerate(focus_chs)
                        if ch in channels and start <= channels.index(ch) <= end]
            if valid_in:
                ax.text(np.mean(valid_in), ax.get_ylim()[1] * 1.04, rname,
                        ha='center', fontsize=8, color='#555555', style='italic')

        fig.savefig(os.path.join(out_dir, 'interpolation_bias_groups.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved interpolation_bias_groups.png")


def plot_bad_channels_vs_graph_metrics(subjects, out_dir):
    """
    B3 — RANSAC bad channel count vs graph theory metrics, by group.

    If RANSAC flags channels because they are spatially unpredictable due to
    dynamic network reconfiguration (hypothesis 2), then subjects with more
    bad channels should also show higher clustering and modularity — both
    signatures of spatially segregated, locally specialised networks that
    produce spatially heterogeneous signals.

    If instead bad channels are pure noise/artifact, no such correlation is
    expected — graph metrics should be independent of bad channel count.

    Metrics: clustering, modularity, efficiency, strength, CPL, betweenness.
    Results averaged across all 15 measure x band combinations then correlated
    with n_bad per subject, separately for ADHD and TDC.
    """
    os.makedirs(out_dir, exist_ok=True)

    from scipy.stats import spearmanr

    METRICS = ['clustering', 'modularity', 'efficiency', 'strength', 'cpl', 'betweenness']
    METRIC_LABELS = {
        'clustering':   'Clustering Coefficient',
        'modularity':   'Modularity',
        'efficiency':   'Global Efficiency',
        'strength':     'Mean Strength',
        'cpl':          'Char. Path Length',
        'betweenness':  'Betweenness Centrality',
    }

    # ── Load graph metrics + n_bad per subject ────────────────────────────────
    rows = []
    for s in subjects:
        sid       = s['subject_id']
        cache_dir = os.path.join(config.RESULTS_ROOT, 'subjects', sid, '.cache')
        pkl_path  = os.path.join(cache_dir, 'graph_results.pkl')
        log       = _parse_run_log(sid)

        if not os.path.exists(pkl_path) or not log:
            continue
        try:
            with open(pkl_path, 'rb') as f:
                g = pickle.load(f)
        except Exception:
            continue

        # Average each metric across all measure x band combos
        metric_vals = {m: [] for m in METRICS}
        for (measure, band), mdict in g.items():
            for m in METRICS:
                v = mdict.get(m)
                if v is not None and np.isfinite(v):
                    metric_vals[m].append(v)

        entry = {
            'sid':   sid,
            'group': s['group'],
            'n_bad': log['n_bad'],
        }
        for m in METRICS:
            entry[m] = float(np.mean(metric_vals[m])) if metric_vals[m] else np.nan

        if not all(np.isnan(entry[m]) for m in METRICS):
            rows.append(entry)

    if not rows:
        print("  Skipping bad_channels_vs_graph_metrics — no data.")
        return

    adhd = [r for r in rows if r['group'] == 'ADHD']
    tdc  = [r for r in rows if r['group'] == 'TDC']
    rng  = np.random.default_rng(42)

    def _boot_spearman(x, y, n_boot=2000):
        x, y = np.array(x), np.array(y)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 5:
            return np.nan, np.nan, np.nan, np.nan
        r_obs, p_obs = spearmanr(x, y)
        idx = np.arange(len(x))
        boot = [spearmanr(x[rng.choice(idx, len(idx), replace=True)],
                          y[rng.choice(idx, len(idx), replace=True)])[0]
                for _ in range(n_boot)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        return r_obs, p_obs, ci_lo, ci_hi

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Scatter grid: n_bad vs each metric (2 rows x 3 cols)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.45, wspace=0.35,
                        left=0.07, right=0.97, bottom=0.08, top=0.92)
    fig.suptitle('B3 — RANSAC Bad Channels vs Graph Theory Metrics\n'
                 '(avg across all 15 measure×band combinations at density=0.25)',
                 fontsize=12, fontweight='bold')

    for ax, metric in zip(axes.flat, METRICS):
        for grp_rows, color, grp_label in [(adhd, ADHD_COLOR, 'ADHD'),
                                            (tdc,  TDC_COLOR,  'TDC')]:
            xs = np.array([r['n_bad'] for r in grp_rows], dtype=float)
            ys = np.array([r[metric]  for r in grp_rows])
            r, p, ci_lo, ci_hi = _boot_spearman(xs, ys)

            jx = rng.uniform(-0.08, 0.08, len(xs))
            ax.scatter(xs + jx, ys, color=color, s=25, alpha=0.65,
                       label=f'{grp_label}  rho={r:.2f}, p={p:.3f}')

            # regression line
            mask = np.isfinite(xs) & np.isfinite(ys)
            if mask.sum() > 2 and len(set(xs[mask])) > 2:
                m_fit, b_fit = np.polyfit(xs[mask], ys[mask], 1)
                x_line = np.linspace(xs[mask].min(), xs[mask].max(), 100)
                ax.plot(x_line, m_fit * x_line + b_fit,
                        color=color, lw=1.6, ls='--', alpha=0.8)

        ax.set_xlabel('RANSAC bad channels (n)', fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.savefig(os.path.join(out_dir, 'bad_channels_vs_graph_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved bad_channels_vs_graph_scatter.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Bootstrap correlation summary bar chart
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.88)

    x_base = np.arange(len(METRICS))
    w      = 0.35

    for gi, (grp_rows, color, grp_label) in enumerate([(adhd, ADHD_COLOR, 'ADHD'),
                                                        (tdc,  TDC_COLOR,  'TDC')]):
        rs, ci_los, ci_his, sigs = [], [], [], []
        for metric in METRICS:
            xs = np.array([r['n_bad']  for r in grp_rows], dtype=float)
            ys = np.array([r[metric]   for r in grp_rows])
            r, p, ci_lo, ci_hi = _boot_spearman(xs, ys)
            rs.append(r)
            ci_los.append(r - ci_lo)
            ci_his.append(ci_hi - r)
            sigs.append('***' if p < 0.001 else ('**' if p < 0.01
                         else ('*' if p < 0.05 else '')))

        xpos = x_base + (gi - 0.5) * w
        ax.bar(xpos, rs, width=w, color=color, alpha=0.75, label=grp_label)
        ci_los_clipped = [max(0, v) for v in ci_los]
        ci_his_clipped = [max(0, v) for v in ci_his]
        ax.errorbar(xpos, rs, yerr=[ci_los_clipped, ci_his_clipped],
                    fmt='none', color='#333333', capsize=4, lw=1.5)
        for xi, (r_val, ci_hi, sig) in enumerate(zip(rs, ci_his, sigs)):
            if sig:
                ax.text(xpos[xi], r_val + ci_hi + 0.02, sig,
                        ha='center', fontsize=10, color='#333333')

    ax.axhline(0, color='#333333', lw=0.9)
    ax.set_xticks(x_base)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9,
                       rotation=15, ha='right')
    ax.set_ylabel('Spearman rho (n_bad vs metric)\nwith 95% bootstrap CI', fontsize=10)
    ax.set_title('B3 — Correlation Summary: Bad Channels vs Graph Metrics by Group\n'
                 'Positive rho = more bad channels → higher metric (supports H2)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(-0.5, 0.6)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    fig.savefig(os.path.join(out_dir, 'bad_channels_vs_graph_corr.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved bad_channels_vs_graph_corr.png")


def plot_kpss_stationarity(subjects, out_dir):
    """
    Distribution of KPSS non-stationarity failure rate (%) per subject by group.
    High failure rates indicate persistent drifts that survived preprocessing.
    Panel 1 — Boxplot + jitter.
    Panel 2 — Histogram overlay (kernel density).
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for s in subjects:
        log = _parse_run_log(s['subject_id'])
        if log and log.get('kpss_fail_pct') is not None:
            rows.append({'group': s['group'], 'kpss_pct': log['kpss_fail_pct']})

    if not rows:
        print("  Skipping kpss_stationarity — no log data.")
        return

    adhd_vals = [r['kpss_pct'] for r in rows if r['group'] == 'ADHD']
    tdc_vals  = [r['kpss_pct'] for r in rows if r['group'] == 'TDC']
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('KPSS Non-Stationarity Rate per Subject', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97, bottom=0.12, top=0.92)

    # ── Panel 1: boxplot + jitter ─────────────────────────────────────────────
    ax = axes[0]
    for vals, x_pos, color, label in [
        (adhd_vals, 1, ADHD_COLOR, f'ADHD (n={len(adhd_vals)})'),
        (tdc_vals,  2, TDC_COLOR,  f'TDC  (n={len(tdc_vals)})'),
    ]:
        if not vals:
            continue
        bp = ax.boxplot([vals], positions=[x_pos], patch_artist=True,
                        widths=0.5, showfliers=False,
                        medianprops=dict(color='#333333', linewidth=1.5))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                   color=color, s=22, alpha=0.7, zorder=3, label=label)
        ax.text(x_pos, max(vals) + 0.3, f'μ={np.mean(vals):.1f}%',
                ha='center', fontsize=8, color=color, fontweight='bold')

    ax.set_xlim(0.3, 2.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ADHD', 'TDC'], fontsize=10)
    ax.set_ylabel('KPSS failure rate (%)', fontsize=10)
    ax.set_title('KPSS Failure Rate\n(% epoch×channel tests)', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    # ── Panel 2: histogram overlay ────────────────────────────────────────────
    ax = axes[1]
    bins = np.linspace(0, max(adhd_vals + tdc_vals) + 1, 20)
    ax.hist(adhd_vals, bins=bins, color=ADHD_COLOR, alpha=0.5,
            label=f'ADHD (n={len(adhd_vals)})')
    ax.hist(tdc_vals,  bins=bins, color=TDC_COLOR,  alpha=0.5,
            label=f'TDC  (n={len(tdc_vals)})')
    ax.axvline(np.mean(adhd_vals), color=ADHD_COLOR, lw=1.5, ls='--', alpha=0.8)
    ax.axvline(np.mean(tdc_vals),  color=TDC_COLOR,  lw=1.5, ls='--', alpha=0.8)
    ax.set_xlabel('KPSS failure rate (%)', fontsize=10)
    ax.set_ylabel('Number of subjects', fontsize=10)
    ax.set_title('Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    fig.savefig(os.path.join(out_dir, 'kpss_stationarity.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved kpss_stationarity.png")


def plot_icalabel_group(subjects, out_dir):
    """
    ICLabel artifact profile — ADHD vs TDC.

    Panel 1 — Artifact burden: total excluded components per subject (boxplot + jitter).
    Panel 2 — Excluded by label: mean # of each artifact class excluded per subject.
    Panel 3 — Class probability ranking: mean probability per ICLabel class
               averaged across all ICs per subject, then group-averaged.
               Reveals systematic artifact loading differences even when no
               single component crosses the exclusion threshold.
    """
    os.makedirs(out_dir, exist_ok=True)
    subjects = _load_ica_reports(subjects)

    valid = [s for s in subjects if s.get('ica_report') is not None]
    if not valid:
        print("  Skipping ICLabel group figure — no ica_report.pkl files found.")
        print("  Re-run pipeline subjects to regenerate them.")
        return

    adhd  = [s for s in valid if s['group'] == 'ADHD']
    tdc   = [s for s in valid if s['group'] == 'TDC']
    n_adhd, n_tdc = len(adhd), len(tdc)

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.38, top=0.86, bottom=0.15, left=0.06, right=0.97)

    # ── Panel 1: Total excluded per subject ──────────────────────────────────
    ax = axes[0]
    adhd_excl = [len(s['ica_report']['excluded']) for s in adhd]
    tdc_excl  = [len(s['ica_report']['excluded']) for s in tdc]

    for vals, x_pos, color, label in [
        (adhd_excl, 1, ADHD_COLOR, f'ADHD (n={n_adhd})'),
        (tdc_excl,  2, TDC_COLOR,  f'TDC  (n={n_tdc})'),
    ]:
        if not vals:
            continue
        bp = ax.boxplot([vals], positions=[x_pos], patch_artist=True,
                        widths=0.5, showfliers=False,
                        medianprops=dict(color='#333333', linewidth=1.5))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                   color=color, s=22, alpha=0.7, zorder=3, label=label)
        ax.text(x_pos, max(vals) + 0.15, f'μ={np.mean(vals):.1f}',
                ha='center', fontsize=8, color=color, fontweight='bold')

    ax.set_xlim(0.3, 2.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ADHD', 'TDC'], fontsize=10)
    ax.set_ylabel('# ICA components excluded', fontsize=9)
    ax.set_title('Artifact Burden\n(total excluded / subject)', fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc='upper right')

    # ── Panel 2: Excluded components by label ────────────────────────────────
    ax = axes[1]

    label_counts = {
        'ADHD': {cls: [] for cls in _ICLABEL_ARTIFACT_CLS},
        'TDC':  {cls: [] for cls in _ICLABEL_ARTIFACT_CLS},
    }
    for grp_key, grp_subjects in [('ADHD', adhd), ('TDC', tdc)]:
        for s in grp_subjects:
            exc_labels = s['ica_report']['excluded_labels']
            for cls in _ICLABEL_ARTIFACT_CLS:
                label_counts[grp_key][cls].append(exc_labels.count(cls))

    short_names = ['muscle', 'eye', 'heart', 'line\nnoise', 'chan\nnoise']
    x = np.arange(len(_ICLABEL_ARTIFACT_CLS))
    w = 0.35

    means_adhd = [np.mean(label_counts['ADHD'][c]) for c in _ICLABEL_ARTIFACT_CLS]
    means_tdc  = [np.mean(label_counts['TDC'][c])  for c in _ICLABEL_ARTIFACT_CLS]

    ax.bar(x - w / 2, means_adhd, w, color=ADHD_COLOR, alpha=0.75,
           label=f'ADHD (n={n_adhd})')
    ax.bar(x + w / 2, means_tdc,  w, color=TDC_COLOR,  alpha=0.75,
           label=f'TDC  (n={n_tdc})')

    # Annotate non-zero bars
    for xi, (va, vt) in enumerate(zip(means_adhd, means_tdc)):
        if va > 0.01:
            ax.text(xi - w / 2, va + 0.01, f'{va:.2f}', ha='center',
                    fontsize=7, color=ADHD_COLOR)
        if vt > 0.01:
            ax.text(xi + w / 2, vt + 0.01, f'{vt:.2f}', ha='center',
                    fontsize=7, color=TDC_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel('Mean components excluded / subject', fontsize=9)
    ax.set_title('Excluded by Label\n(mean per subject per class)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    # ── Panel 3: Mean class probability ranking ──────────────────────────────
    ax = axes[2]

    mean_probas_adhd = []
    mean_probas_tdc  = []
    for s in adhd:
        probas = s['ica_report']['probas']   # (n_components, 7)
        mean_probas_adhd.append(probas.mean(axis=0))
    for s in tdc:
        probas = s['ica_report']['probas']
        mean_probas_tdc.append(probas.mean(axis=0))

    x = np.arange(7)
    w = 0.35
    gm_adhd = np.mean(mean_probas_adhd, axis=0) if mean_probas_adhd else np.zeros(7)
    gm_tdc  = np.mean(mean_probas_tdc,  axis=0) if mean_probas_tdc  else np.zeros(7)

    bars_adhd = ax.bar(x - w / 2, gm_adhd, w, color=ADHD_COLOR, alpha=0.75,
                       label=f'ADHD (n={n_adhd})')
    bars_tdc  = ax.bar(x + w / 2, gm_tdc,  w, color=TDC_COLOR,  alpha=0.75,
                       label=f'TDC  (n={n_tdc})')

    for xi, (va, vt) in enumerate(zip(gm_adhd, gm_tdc)):
        if va > 0.005:
            ax.text(xi - w / 2, va + 0.005, f'{va:.3f}', ha='center',
                    fontsize=6, color=ADHD_COLOR)
        if vt > 0.005:
            ax.text(xi + w / 2, vt + 0.005, f'{vt:.3f}', ha='center',
                    fontsize=6, color=TDC_COLOR)

    class_display = ['brain', 'muscle', 'eye', 'heart', 'line\nnoise', 'chan\nnoise', 'other']
    ax.set_xticks(x)
    ax.set_xticklabels(class_display, fontsize=8)
    ax.set_ylabel('Mean probability (averaged\nacross all ICA components)', fontsize=9)
    ax.set_title('Class Probability Ranking\n(mean across all ICs per subject)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.set_ylim(0, min(1.05, max(gm_adhd.max(), gm_tdc.max()) * 1.25 + 0.05))
    # Mark exclusion threshold on artifact classes
    ax.axhline(config.ICLABEL_THRESHOLD, color='#888888', linewidth=0.9,
               linestyle='--', alpha=0.6, label=f'threshold ({config.ICLABEL_THRESHOLD})')
    ax.legend(fontsize=7)

    n_valid_adhd = sum(1 for s in adhd if s.get('ica_report'))
    n_valid_tdc  = sum(1 for s in tdc  if s.get('ica_report'))
    fig.suptitle(
        f'ICLabel Artifact Profile — ADHD (n={n_valid_adhd}) vs TDC (n={n_valid_tdc})\n'
        f'Threshold = {config.ICLABEL_THRESHOLD}  |  '
        f'Keep classes: {", ".join(config.ICLABEL_KEEP)}  |  '
        f'{config.ICA_N_COMPONENTS} ICA components per subject',
        fontsize=11,
    )

    fname = 'icalabel_group.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_combined_features(subjects, figures_root):
    """Save all subjects' betti_features into one dict keyed by subject_id."""
    combined = {s['subject_id']: s['features'] for s in subjects}
    path = os.path.normpath(os.path.join(figures_root, '..', 'betti_features_all.pkl'))
    with open(path, 'wb') as f:
        pickle.dump(combined, f)
    print(f"  Saved betti_features_all.pkl  ({len(combined)} subjects)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_group(force=False):
    figures_root  = os.path.join(config.RESULTS_ROOT, 'group', 'figures')
    no_filter_dir = os.path.join(figures_root, 'no_filter')
    sl_dir        = os.path.join(figures_root, 'surface_laplacian')
    overview_dir  = os.path.join(figures_root, 'overview')

    # Per-pipeline subfolders
    for base in [no_filter_dir, sl_dir]:
        for sub in ['auc', 'betti', 'density_sweep', 'edge_dist',
                    'graph_metrics', 'pca_umap', 'topomaps']:
            os.makedirs(os.path.join(base, sub), exist_ok=True)

    # Overview subfolders
    for sub in ['preprocessing',
                'density_sweep/combined', 'edge_dist/combined',
                'graph_metrics/combined', 'pca_umap/combined',
                'pipeline_comparison']:
        os.makedirs(os.path.join(overview_dir, sub), exist_ok=True)

    preprocessing_dir = os.path.join(overview_dir, 'preprocessing')

    print("\nLoading subject data...")
    subjects, skipped = load_all_subjects()

    if skipped:
        print(f"  Skipped {len(skipped)} subject(s) (not yet processed): {skipped}")

    if not subjects:
        print("  No completed subjects found. Run pipeline_subject.py first.")
        return

    n_adhd = sum(1 for s in subjects if s['group'] == 'ADHD')
    n_tdc  = sum(1 for s in subjects if s['group'] == 'TDC')
    print(f"  Loaded {len(subjects)} subjects  (ADHD={n_adhd}, TDC={n_tdc})")

    if n_adhd == 0 or n_tdc == 0:
        print("  WARNING: Only one group present — group comparison figures will be partial.")

    # ── Aggregate pkl ─────────────────────────────────────────────────────────
    print("\nSaving combined features...")
    save_combined_features(subjects, figures_root)

    # ── Check Surface Laplacian availability ──────────────────────────────────
    n_sl = sum(1 for s in subjects if s['features_sl'] is not None)
    print(f"  Surface Laplacian features available for {n_sl}/{len(subjects)} subjects")

    # ── Preprocessing overview ─────────────────────────────────────────────────
    print("\nGenerating preprocessing overview figures...")
    cohort_dir   = os.path.join(preprocessing_dir, 'cohort')
    artifact_dir = os.path.join(preprocessing_dir, 'artifact_metrics')
    hyp_a_dir    = os.path.join(preprocessing_dir, 'hypothesis_A')
    hyp_b_dir    = os.path.join(preprocessing_dir, 'hypothesis_B')
    for d in [cohort_dir, artifact_dir, hyp_a_dir, hyp_b_dir]:
        os.makedirs(d, exist_ok=True)

    # cohort
    plot_group_counts(subjects, cohort_dir)
    generate_preprocessing_summary(subjects, skipped, cohort_dir)

    # artifact metrics
    plot_recording_length(subjects, artifact_dir)
    plot_epoch_retention(subjects, artifact_dir)
    plot_bad_channel_map(subjects, artifact_dir)
    plot_kpss_stationarity(subjects, artifact_dir)
    plot_icalabel_group(subjects, artifact_dir)

    # hypothesis A — recording length instability
    plot_length_vs_bad_channels(subjects, hyp_a_dir)
    plot_length_vs_ica(subjects, hyp_a_dir)
    plot_length_vs_drop_rate(subjects, hyp_a_dir)

    # hypothesis B — network dynamics / interpolation bias
    plot_edge_weight_variance(subjects, hyp_b_dir)
    plot_interpolation_connectivity_bias(subjects, hyp_b_dir)
    plot_bad_channels_vs_graph_metrics(subjects, hyp_b_dir)

    print("\nGenerating graph theory metrics figures (no filter)...")
    plot_graph_metrics_group(subjects, os.path.join(no_filter_dir, 'graph_metrics'), pipeline='no_filter')
    if n_sl > 0:
        print("\nGenerating graph theory metrics figures (surface laplacian)...")
        plot_graph_metrics_group(subjects, os.path.join(sl_dir, 'graph_metrics'), pipeline='surface_laplacian')
        print("\nGenerating graph theory metrics combined figures (no filter + surface laplacian)...")
        plot_graph_metrics_combined(subjects, os.path.join(overview_dir, 'graph_metrics', 'combined'))

    print("\nGenerating density sweep figures (no filter)...")
    plot_density_sweep_group(subjects, os.path.join(no_filter_dir, 'density_sweep'), pipeline='no_filter')
    if n_sl > 0:
        print("\nGenerating density sweep figures (surface laplacian)...")
        plot_density_sweep_group(subjects, os.path.join(sl_dir, 'density_sweep'), pipeline='surface_laplacian')
        print("\nGenerating density sweep combined figures (no filter + surface laplacian)...")
        plot_density_sweep_combined(subjects, os.path.join(overview_dir, 'density_sweep', 'combined'))

    print("\nGenerating pipeline comparison figures (no filter vs surface laplacian)...")
    plot_pipeline_comparison(subjects, os.path.join(overview_dir, 'pipeline_comparison'))

    print("\nGenerating PCA / UMAP figures (no filter)...")
    plot_pca_umap(subjects, os.path.join(no_filter_dir, 'pca_umap'), pipeline='no_filter')
    if n_sl > 0:
        print("\nGenerating PCA / UMAP figures (surface laplacian)...")
        plot_pca_umap(subjects, os.path.join(sl_dir, 'pca_umap'), pipeline='surface_laplacian')
        print("\nGenerating PCA combined figures (no filter + surface laplacian)...")
        plot_pca_combined(subjects, os.path.join(overview_dir, 'pca_umap', 'combined'))

    # ── No filter (no spatial filter) ────────────────────────────────────────
    print("\nGenerating figures — no_filter/ (no spatial filter)...")
    plot_group_betti_curves(subjects, os.path.join(no_filter_dir, 'betti'), pipeline='no_filter')
    plot_auc_distributions(subjects, os.path.join(no_filter_dir, 'auc'), pipeline='no_filter')
    print("Running statistics — no_filter/...")
    run_group_statistics(subjects, no_filter_dir, pipeline='no_filter')

    # ── Surface Laplacian ─────────────────────────────────────────────────────
    if n_sl > 0:
        print("\nGenerating figures — surface_laplacian/...")
        plot_group_betti_curves(subjects, os.path.join(sl_dir, 'betti'), pipeline='surface_laplacian')
        plot_auc_distributions(subjects, os.path.join(sl_dir, 'auc'), pipeline='surface_laplacian')
        print("Running statistics — surface_laplacian/...")
        run_group_statistics(subjects, sl_dir, pipeline='surface_laplacian')
    else:
        print("\nSkipping surface_laplacian/ figures — no subjects have betti_features_csd.pkl yet.")
        print("  Force-rerun subjects to generate Surface Laplacian features.")

    print(f"\nGroup analysis complete.")
    print(f"  results/group/figures/")
    print(f"    no_filter/          — no spatial filter ({n_adhd} ADHD, {n_tdc} TDC)")
    if n_sl > 0:
        print(f"    surface_laplacian/  — surface Laplacian ({n_sl} subjects with SL)")
    print(f"    overview/           — group counts")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run group-level analysis.')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate outputs even if they already exist')
    args = parser.parse_args()
    run_group(force=args.force)
