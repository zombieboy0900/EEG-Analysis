"""
ransac_diagnostics.py — Diagnose RANSAC bad-channel detection across all subjects.

RANSAC flags channel i if the fraction of 1-second epochs where
    corr(predicted_i, actual_i) < 0.75
exceeds min_frac (default 0.01).  Low correlation = channel signal is NOT
predictable from neighbours via spherical spline interpolation.

Two analyses:

  --all-channels   For every channel, compare frac_below (flagged vs clean
                   subjects) and show how far flagged channels exceed min_frac.
                   Saves:
                     ransac_frac_below_threshold.png
                     ransac_threshold_margin.png

  --psd            For focus channels Fz, Cz, F7: compare PSD during bad epochs
                   (RANSAC corr < 0.75) vs good epochs, split by group.
                   Also shows overall group-average PSD as a baseline.
                   Saves:
                     ransac_psd_bad_vs_good_{Fz,Cz,F7}.png
"""

import os, sys, argparse, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import mne
from mne.io import RawArray
from autoreject import Ransac

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

REGION_ORDER = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7',  'C3',  'Cz', 'C4', 'T8',
    'P7',  'P3',  'Pz', 'P4', 'P8',
    'O1',  'O2',
]

THRESHOLD = 0.75   # per-epoch corr threshold (RANSAC min_corr)
MIN_FRAC  = 0.01   # autoreject default: flag if >1% of epochs are bad


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_all_sids():
    import csv
    metadata = {}
    with open(config.METADATA_CSV, newline='') as f:
        for row in csv.DictReader(f):
            metadata[row['subject_id'].strip()] = row['group'].strip()
    sids = []
    for sid, grp in metadata.items():
        mat = os.path.join(config.RAW_DIR, sid + '.mat')
        pkl = os.path.join(config.RESULTS_ROOT, 'subjects', sid, '.cache', 'bad_channels.pkl')
        if os.path.exists(mat) and os.path.exists(pkl):
            sids.append((sid, grp))
    return sids


def _run_ransac(subject_id):
    """
    Load .mat, apply notch + bandpass, make 1-second epochs, fit RANSAC.
    Returns (rsc, raw, epochs).
      rsc.corr_  : (n_epochs, n_channels) per-epoch prediction correlations
      rsc.bad_chs_: list of channels flagged bad
    """
    mat_path = os.path.join(config.RAW_DIR, subject_id + '.mat')
    mat = scipy.io.loadmat(mat_path)

    key  = subject_id if subject_id in mat else [k for k in mat if not k.startswith('_')][0]
    data = mat[key]
    if data.shape[0] > data.shape[1]:
        data = data.T

    ch_names_generic = [f'EEG{i:03d}' for i in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names_generic, sfreq=config.SFREQ, ch_types='eeg')
    raw  = RawArray(data * 1e-6, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    if len(raw.ch_names) == len(config.STANDARD_19):
        mapping = {raw.ch_names[i]: config.STANDARD_19[i]
                   for i in range(len(config.STANDARD_19))}
        raw.rename_channels(mapping)
        raw.set_montage(montage)

    raw.notch_filter(freqs=60.0, verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, phase='zero', fir_design='firwin', verbose=False)

    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose=False)
    rsc    = Ransac(n_jobs=1, verbose=False)
    rsc.fit(epochs)

    return rsc, raw, epochs


# ── Analysis 1: all-channel frac_below ────────────────────────────────────────

def run_all_channels(out_dir=None):
    """
    Run RANSAC on every subject. For every channel compare frac_below between
    flagged and non-flagged subjects, split by group.  Also show how far above
    MIN_FRAC the flagged channels sit (the actual excess that drives the decision).

    Produces:
      ransac_frac_below_threshold.png  — per-channel flagged vs clean scatter
      ransac_threshold_margin.png      — excess frac_below for flagged channels
    """
    if out_dir is None:
        out_dir = os.path.join(config.RESULTS_ROOT, 'group', 'figures',
                               'overview', 'preprocessing', 'hypothesis_B')
    os.makedirs(out_dir, exist_ok=True)

    all_sids = _load_all_sids()
    print(f"Running RANSAC on all {len(all_sids)} subjects...")

    from collections import defaultdict
    ch_records = defaultdict(list)   # ch -> [{frac_below, flagged, group}]

    for i, (sid, grp) in enumerate(all_sids):
        print(f"  [{i+1}/{len(all_sids)}] {sid}...")
        try:
            rsc, raw, _ = _run_ransac(sid)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

        corr_arr   = rsc.corr_
        flagged    = set(rsc.bad_chs_)
        frac_below = np.mean(corr_arr < THRESHOLD, axis=0)

        for j, ch in enumerate(raw.ch_names):
            ch_records[ch].append({
                'frac_below': float(frac_below[j]),
                'flagged':    ch in flagged,
                'group':      grp,
            })

    channels = [ch for ch in REGION_ORDER if ch in ch_records]

    # ── Figure 1: frac_below flagged vs clean per channel ─────────────────────
    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    fig.suptitle(
        'RANSAC: Fraction of Epochs Below Threshold — Flagged vs Not-Flagged\n'
        'Every channel independently — removes selection bias\n'
        'ADHD (red) | TDC (blue)   |   dashed = min_frac (0.01)',
        fontsize=12, fontweight='bold',
    )
    fig.subplots_adjust(hspace=0.6, wspace=0.4,
                        left=0.05, right=0.98, bottom=0.05, top=0.90)

    rng_j   = np.random.default_rng(42)
    ax_flat = axes.flat

    for ch in channels:
        ax   = next(ax_flat, None)
        if ax is None:
            break
        recs = ch_records[ch]

        for grp, color_flag, color_ok, x_flag, x_ok in [
            ('ADHD', '#c0392b', '#f1948a', 0, 1),
            ('TDC',  '#1a6fc4', '#7fb8e8', 2, 3),
        ]:
            for vals, x_pos, color in [
                ([r['frac_below'] for r in recs if r['flagged'] and r['group'] == grp],
                 x_flag, color_flag),
                ([r['frac_below'] for r in recs if not r['flagged'] and r['group'] == grp],
                 x_ok, color_ok),
            ]:
                if not vals:
                    continue
                jx = rng_j.uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full(len(vals), x_pos) + jx, vals,
                           color=color, s=10, alpha=0.7, zorder=3)
                ax.plot([x_pos - 0.25, x_pos + 0.25],
                        [np.mean(vals), np.mean(vals)],
                        color=color, lw=2.0)

        ax.axhline(MIN_FRAC, color='#333333', lw=0.9, ls='--', alpha=0.7)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['A\nflag', 'A\nok', 'T\nflag', 'T\nok'], fontsize=6)
        ax.set_title(ch, fontsize=10, fontweight='bold', pad=2)
        ax.set_ylabel('Frac. epochs < 0.75', fontsize=5)
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.3)

    for ax in ax_flat:
        ax.set_visible(False)

    fig.savefig(os.path.join(out_dir, 'ransac_frac_below_threshold.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved ransac_frac_below_threshold.png")

    # ── Figure 2: excess frac_below above MIN_FRAC for flagged channels ────────
    margin_data = {}
    for ch in channels:
        for grp in ['ADHD', 'TDC']:
            vals = [r['frac_below'] for r in ch_records[ch]
                    if r['flagged'] and r['group'] == grp]
            if len(vals) >= 3:
                margin_data[(ch, grp)] = [v - MIN_FRAC for v in vals]

    plot_chs = sorted(set(ch for ch, _ in margin_data),
                      key=lambda c: channels.index(c) if c in channels else 99)

    if plot_chs:
        fig, ax = plt.subplots(figsize=(16, 5))
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.88)

        x, w = np.arange(len(plot_chs)), 0.38
        for grp, color, offset in [('ADHD', '#c0392b', -w/2), ('TDC', '#1a6fc4', w/2)]:
            means, sems, ns = [], [], []
            for ch in plot_chs:
                vals = margin_data.get((ch, grp), [])
                means.append(np.mean(vals) if vals else np.nan)
                sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                ns.append(len(vals))

            ax.bar(x + offset, means, w, color=color, alpha=0.75, label=grp)
            ax.errorbar(x + offset, means, yerr=sems,
                        fmt='none', color='#333333', capsize=3, lw=1.2)
            for xi, (m, n) in enumerate(zip(means, ns)):
                if not np.isnan(m) and n > 0:
                    y_txt = (m + sems[xi] + 0.005) if m >= 0 else (m - sems[xi] - 0.015)
                    ax.text(x[xi] + offset, y_txt, f'n={n}',
                            ha='center', fontsize=6.5, color=color)

        ax.axhline(0, color='#333333', lw=1.0, ls='--', alpha=0.6,
                   label=f'min_frac ({MIN_FRAC})')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_chs, fontsize=9)
        ax.set_ylabel('Excess frac_below above min_frac\n(larger = more bad epochs)',
                      fontsize=9)
        ax.set_title(
            'How Many Bad Epochs Drive Each Flagged Channel?\n'
            'Excess fraction of epochs below 0.75 (the actual RANSAC decision metric)',
            fontsize=11, fontweight='bold',
        )
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

        fig.savefig(os.path.join(out_dir, 'ransac_threshold_margin.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved ransac_threshold_margin.png")


# ── Analysis 2: PSD bad vs good epochs ────────────────────────────────────────

def plot_psd_by_frac_below(out_dir=None):
    """
    For focus channels Fz, Cz (TDC interest) and F7 (ADHD interest), compare:

      Panel A — PSD during bad epochs (RANSAC corr < 0.75 at that channel)
                vs good epochs, averaged across all subjects per group.
                Artifact  → bad epochs have elevated broadband / gamma (flat)
                Real signal → bad epochs same 1/f shape, just higher amplitude

      Panel B — Overall group-average PSD (all epochs).
                Baseline: are ADHD and TDC just spectrally different overall?

    Produces:
      ransac_psd_bad_vs_good_{Fz,Cz,F7}.png
    """
    from scipy.signal import welch
    from collections import defaultdict

    if out_dir is None:
        out_dir = os.path.join(config.RESULTS_ROOT, 'group', 'figures',
                               'overview', 'preprocessing', 'hypothesis_B')
    os.makedirs(out_dir, exist_ok=True)

    FOCUS_CHS = ['Fz', 'Cz', 'F7']
    NPERSEG   = config.SFREQ   # 1-second window → 1 Hz resolution

    all_sids = _load_all_sids()
    print(f"Computing epoch PSDs for {len(all_sids)} subjects...")

    # ch -> group -> 'bad'/'good'/'all' -> list of per-subject mean PSDs
    psd_store = {ch: {'ADHD': {'bad': [], 'good': [], 'all': []},
                      'TDC':  {'bad': [], 'good': [], 'all': []}}
                 for ch in FOCUS_CHS}
    freqs_out = None

    for i, (sid, grp) in enumerate(all_sids):
        print(f"  [{i+1}/{len(all_sids)}] {sid}...")
        try:
            rsc, raw, epochs = _run_ransac(sid)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

        ch_names = raw.ch_names
        corr_arr = rsc.corr_           # (n_epochs, n_ch)
        ep_data  = epochs.get_data()   # (n_epochs, n_ch, n_times)

        def _mean_psd(epoch_mat):
            if len(epoch_mat) == 0:
                return None
            psds = []
            for ep in epoch_mat:
                f, p = welch(ep, fs=config.SFREQ, nperseg=NPERSEG, scaling='density')
                psds.append(p)
            return np.mean(psds, axis=0), f

        for ch in FOCUS_CHS:
            if ch not in ch_names:
                continue
            ci      = ch_names.index(ch)
            ch_corr = corr_arr[:, ci]
            ch_ep   = ep_data[:, ci, :]

            bad_mask  = ch_corr < THRESHOLD
            good_mask = ~bad_mask

            for key, mask in [('bad', bad_mask), ('good', good_mask), ('all', slice(None))]:
                result = _mean_psd(ch_ep[mask])
                if result is not None:
                    psd_store[ch][grp][key].append(result[0])
                    freqs_out = result[1]

    if freqs_out is None:
        print("No data collected — aborting.")
        return

    freq_mask  = (freqs_out >= 1) & (freqs_out <= 45)
    freqs_plot = freqs_out[freq_mask]

    BAND_REGIONS = [
        ('delta', 1,  4,  '#aaaaaa'),
        ('theta', 4,  8,  '#bbbbff'),
        ('alpha', 8,  13, '#aaffaa'),
        ('beta',  13, 30, '#ffddaa'),
        ('gamma', 30, 45, '#ffaaaa'),
    ]

    def _plot_line(ax, psds_list, color, ls, label):
        if not psds_list:
            return
        mat      = np.array([p[freq_mask] for p in psds_list])
        mean_psd = np.mean(mat, axis=0)
        sem_psd  = np.std(mat, axis=0) / np.sqrt(len(mat))
        ax.semilogy(freqs_plot, mean_psd, color=color, ls=ls, lw=2.0,
                    label=f'{label} (n={len(mat)})')
        ax.fill_between(freqs_plot,
                        np.maximum(mean_psd - sem_psd, 1e-30),
                        mean_psd + sem_psd,
                        color=color, alpha=0.12)

    def _add_bands(ax):
        for band, flo, fhi, bc in BAND_REGIONS:
            ax.axvspan(flo, fhi, alpha=0.07, color=bc)
        for band, flo, fhi, _ in BAND_REGIONS:
            ax.text((flo + fhi) / 2, 0.98, band, ha='center', fontsize=7,
                    color='#555555', style='italic',
                    transform=ax.get_xaxis_transform())

    for ch in FOCUS_CHS:
        store = psd_store[ch]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12,
                            top=0.85, wspace=0.08)
        fig.suptitle(
            f'PSD at {ch} — Bad epochs vs Good epochs  |  ADHD (red) vs TDC (blue)\n'
            f'Artifact: bad epochs elevated broadband/gamma   '
            f'Real signal: bad epochs same 1/f shape as good',
            fontsize=11, fontweight='bold',
        )

        ax = axes[0]
        _plot_line(ax, store['ADHD']['bad'],  '#7b1010', '-',  'ADHD — bad epochs')
        _plot_line(ax, store['ADHD']['good'], '#f1948a', '--', 'ADHD — good epochs')
        _plot_line(ax, store['TDC']['bad'],   '#0d4f8b', '-',  'TDC  — bad epochs')
        _plot_line(ax, store['TDC']['good'],  '#7fb8e8', '--', 'TDC  — good epochs')
        ax.set_title('Panel A: Bad vs Good Epochs\n(split by RANSAC corr < 0.75)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('PSD (V²/Hz, log scale)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(1, 45)
        ax.grid(True, which='both', linewidth=0.3, alpha=0.4)
        _add_bands(ax)

        ax = axes[1]
        _plot_line(ax, store['ADHD']['all'], '#c0392b', '-', 'ADHD — all epochs')
        _plot_line(ax, store['TDC']['all'],  '#1a6fc4', '-', 'TDC  — all epochs')
        ax.set_title('Panel B: Overall Group Average\n(all epochs combined)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(1, 45)
        ax.grid(True, which='both', linewidth=0.3, alpha=0.4)
        _add_bands(ax)

        fname = f'ransac_psd_bad_vs_good_{ch}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RANSAC bad-channel diagnostics')
    parser.add_argument('--all-channels', action='store_true',
                        help='frac_below per channel: flagged vs clean, all subjects')
    parser.add_argument('--psd', action='store_true',
                        help='PSD: bad vs good epochs at Fz, Cz, F7')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    if args.all_channels:
        run_all_channels(out_dir=args.out_dir)
    elif args.psd:
        plot_psd_by_frac_below(out_dir=args.out_dir)
    else:
        parser.print_help()
