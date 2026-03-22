"""
compare_runs.py — Evaluate the impact of preprocessing changes.

Workflow
--------
1. Before making a code change, snapshot the current results:
       python pipeline/compare_runs.py --snapshot baseline

2. Edit the code, rerun a test subset (10 subjects recommended):
       python pipeline/run_all.py --subjects v10p v12p v14p v15p v18p v41p v42p v43p v44p v45p --force

3. Compare new results against the snapshot:
       python pipeline/compare_runs.py --compare baseline

The snapshot stores a lightweight CSV summary (no large pkl files copied).
The compare report shows:
  - Preprocessing changes (epochs, bad channels, ICA exclusions)
  - Connectivity matrix correlation per subject (measures how much the
    raw EEG connectivity changed — proxy for how different the data looks)
  - Betti AUC changes per feature
  - Effect size (Cohen's d) shift for ADHD vs TDC on each feature
"""

import os
import sys
import csv
import pickle
import argparse
import datetime
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

SNAPSHOTS_DIR = os.path.join(config.RESULTS_ROOT, 'snapshots')
METADATA_CSV  = config.METADATA_CSV

# ── Suggested test set: varied bad-channel counts and both groups ──────────────
# Edit this to whatever subjects you want to use for quick evaluation runs.
DEFAULT_TEST_SUBJECTS = [
    'v10p', 'v12p', 'v14p', 'v18p', 'v19p',   # ADHD
    'v41p', 'v42p', 'v43p', 'v44p', 'v45p',   # TDC
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_group_map():
    """Return {subject_id: 'ADHD'|'TDC'} from metadata.csv."""
    gmap = {}
    with open(METADATA_CSV, newline='') as f:
        for row in csv.DictReader(f):
            gmap[row['subject_id'].strip()] = row['group'].strip()
    return gmap


def _subject_dir(sid):
    return os.path.join(config.RESULTS_ROOT, 'subjects', sid)


def _cache_dir(sid):
    return os.path.join(_subject_dir(sid), '.cache')


def _load_pkl_safe(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _parse_epoch_count(sid):
    """Extract clean epoch count from run.log (last Preprocessing log line)."""
    log_path = os.path.join(_subject_dir(sid), 'run.log')
    if not os.path.exists(log_path):
        return None
    count = None
    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'after AutoReject' in line:
                # "Epochs: 73 -> 65 after AutoReject (8 dropped)"
                parts = line.split('->')
                if len(parts) >= 2:
                    try:
                        count = int(parts[1].split()[0])
                    except (ValueError, IndexError):
                        pass
    return count


def _extract_subject_summary(sid):
    """
    Pull key per-subject metrics from the current results on disk.
    Returns a dict or None if subject has no results.
    """
    bf_path  = os.path.join(_subject_dir(sid), 'betti_features.pkl')
    if not os.path.exists(bf_path):
        return None

    betti_features = _load_pkl_safe(bf_path)
    bad_chs        = _load_pkl_safe(os.path.join(_cache_dir(sid), 'bad_channels.pkl'))
    ica_report     = _load_pkl_safe(os.path.join(_subject_dir(sid), 'ica_report.pkl'))
    n_epochs       = _parse_epoch_count(sid)

    row = {
        'subject_id':     sid,
        'n_epochs':        n_epochs,
        'n_bad_channels':  len(bad_chs) if bad_chs is not None else None,
        'n_ica_excluded':  len(ica_report['excluded']) if ica_report else None,
    }

    # AUC features for all (measure, band) combinations
    if betti_features:
        for (measure, band), feats in betti_features.items():
            prefix = f"{measure[:3]}_{band[:3]}"
            row[f'{prefix}_auc_b0']  = feats.get('auc_b0',  None)
            row[f'{prefix}_auc_b1']  = feats.get('auc_b1',  None)
            row[f'{prefix}_slope_b0'] = feats.get('slope_b0', None)

    return row


def _load_conn_upper(sid):
    """
    Load connectivity matrices for a subject and return {key: upper_triangle_vector}.
    Used to compute matrix-level similarity between runs.
    """
    result = {}
    npz_path = os.path.join(_cache_dir(sid), 'conn_matrices.npz')
    if not os.path.exists(npz_path):
        return result
    try:
        data = np.load(npz_path)
        for k in data.files:
            mat = data[k]
            n   = mat.shape[0]
            idx = np.triu_indices(n, k=1)
            result[k] = mat[idx]
    except Exception:
        pass
    return result


def _cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════

def snapshot(name, subjects_filter=None):
    """
    Save a lightweight summary CSV of current results.
    Does NOT copy large pkl/fif files — only extracts key metrics.
    """
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    snap_dir  = os.path.join(SNAPSHOTS_DIR, name)
    os.makedirs(snap_dir, exist_ok=True)

    gmap = _load_group_map()

    # Find all subjects with results
    all_sids = [sid for sid in gmap
                if os.path.exists(os.path.join(_subject_dir(sid), 'betti_features.pkl'))]

    if subjects_filter:
        all_sids = [s for s in all_sids if s in subjects_filter]

    rows = []
    for sid in sorted(all_sids):
        row = _extract_subject_summary(sid)
        if row:
            row['group'] = gmap.get(sid, '')
            rows.append(row)

    if not rows:
        print(f"  No subjects with results found. Nothing to snapshot.")
        return

    # Determine all columns (union of keys)
    all_cols = ['subject_id', 'group', 'n_epochs', 'n_bad_channels', 'n_ica_excluded']
    extra_cols = sorted({k for r in rows for k in r if k not in all_cols})
    all_cols += extra_cols

    csv_path = os.path.join(snap_dir, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, '') for c in all_cols})

    # Also snapshot connectivity matrices (upper triangles only — much smaller)
    conn_snap = {}
    for row in rows:
        sid = row['subject_id']
        ut  = _load_conn_upper(sid)
        if ut:
            conn_snap[sid] = ut
    if conn_snap:
        np.save(os.path.join(snap_dir, 'conn_upper.npy'), conn_snap)

    # Metadata
    meta_path = os.path.join(snap_dir, 'meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"Snapshot: {name}\n")
        f.write(f"Created:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Subjects: {len(rows)}\n")
        f.write(f"ADHD:     {sum(1 for r in rows if r['group'] == 'ADHD')}\n")
        f.write(f"TDC:      {sum(1 for r in rows if r['group'] == 'TDC')}\n")

    print(f"Snapshot '{name}' saved: {len(rows)} subjects")
    print(f"  {snap_dir}/summary.csv")
    if conn_snap:
        print(f"  {snap_dir}/conn_upper.npy  ({len(conn_snap)} subjects)")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE
# ══════════════════════════════════════════════════════════════════════════════

def _load_snapshot_csv(name):
    """Return list of dicts from snapshot summary.csv."""
    path = os.path.join(SNAPSHOTS_DIR, name, 'summary.csv')
    if not os.path.exists(path):
        print(f"ERROR: Snapshot '{name}' not found at {path}")
        sys.exit(1)
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _load_snapshot_conn(name):
    """Return connectivity upper-triangle dict from snapshot, or {}."""
    path = os.path.join(SNAPSHOTS_DIR, name, 'conn_upper.npy')
    if not os.path.exists(path):
        return {}
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception:
        return {}


def compare(baseline_name, subjects_filter=None):
    """
    Compare current results against a saved snapshot.
    Prints a structured report showing what changed and by how much.
    """
    gmap = _load_group_map()

    # ── Load baseline ────────────────────────────────────────────────────────
    base_rows   = _load_snapshot_csv(baseline_name)
    base_by_sid = {r['subject_id']: r for r in base_rows}
    base_conn   = _load_snapshot_conn(baseline_name)

    # ── Load current ─────────────────────────────────────────────────────────
    sids = list(base_by_sid.keys())
    if subjects_filter:
        sids = [s for s in sids if s in subjects_filter]

    curr_rows = []
    for sid in sids:
        row = _extract_subject_summary(sid)
        if row:
            row['group'] = gmap.get(sid, '')
            curr_rows.append(row)

    curr_by_sid = {r['subject_id']: r for r in curr_rows}

    common = sorted(set(base_by_sid) & set(curr_by_sid))
    if subjects_filter:
        common = [s for s in common if s in subjects_filter]

    if not common:
        print("No subjects found in both baseline and current results.")
        return

    # ── Section 1: Preprocessing metrics ────────────────────────────────────
    print('\n' + '=' * 70)
    print('PREPROCESSING METRICS')
    print('=' * 70)
    print(f"{'Subject':<10} {'Group':<6} {'Epochs':>14} {'Bad ch':>12} {'ICA excl':>12}")
    print(f"{'':10} {'':6} {'base → new':>14} {'base → new':>12} {'base → new':>12}")
    print('-' * 70)

    ep_deltas, bc_deltas, ica_deltas = [], [], []

    for sid in common:
        b = base_by_sid[sid]
        c = curr_by_sid[sid]
        grp = gmap.get(sid, '?')

        def fmt_delta(bval, cval, field):
            try:
                bv = int(float(bval)) if bval not in ('', None) else None
                cv = int(float(cval)) if cval not in ('', None) else None
                if bv is None or cv is None:
                    return f'{'?':>5} → {'?':>5}'
                delta = cv - bv
                sign  = '+' if delta > 0 else ''
                return f'{bv:>5} → {cv:>5} ({sign}{delta})'
            except Exception:
                return '?'

        ep_str  = fmt_delta(b.get('n_epochs'),       c.get('n_epochs'),       'n_epochs')
        bc_str  = fmt_delta(b.get('n_bad_channels'), c.get('n_bad_channels'), 'n_bad_channels')
        ica_str = fmt_delta(b.get('n_ica_excluded'), c.get('n_ica_excluded'), 'n_ica_excluded')

        print(f"{sid:<10} {grp:<6} {ep_str:>14} {bc_str:>12} {ica_str:>12}")

        for store, bkey, ckey in [
            (ep_deltas,  b.get('n_epochs'),       c.get('n_epochs')),
            (bc_deltas,  b.get('n_bad_channels'), c.get('n_bad_channels')),
            (ica_deltas, b.get('n_ica_excluded'), c.get('n_ica_excluded')),
        ]:
            try:
                store.append(float(ckey) - float(bkey))
            except (TypeError, ValueError):
                pass

    if ep_deltas:
        print(f"\n  Mean epoch change:     {np.mean(ep_deltas):+.1f}  "
              f"(min {min(ep_deltas):+.0f}, max {max(ep_deltas):+.0f})")
    if ica_deltas:
        print(f"  Mean ICA excl change:  {np.mean(ica_deltas):+.2f}")

    # ── Section 2: Connectivity matrix correlation ───────────────────────────
    print('\n' + '=' * 70)
    print('CONNECTIVITY MATRIX SIMILARITY  (r=1.0 → identical, r<0.95 → notable change)')
    print('=' * 70)

    curr_conn_all = {}
    for sid in common:
        curr_conn_all[sid] = _load_conn_upper(sid)

    # Correlations per key across subjects
    all_keys = set()
    for sid in common:
        all_keys.update(base_conn.get(sid, {}).keys())
        all_keys.update(curr_conn_all.get(sid, {}).keys())

    if not base_conn:
        print("  No connectivity data in baseline snapshot.")
    else:
        key_corrs = {}
        for k in sorted(all_keys):
            pairs = []
            for sid in common:
                bv = base_conn.get(sid, {}).get(k)
                cv = curr_conn_all.get(sid, {}).get(k)
                if bv is not None and cv is not None and len(bv) == len(cv):
                    pairs.append((bv, cv))
            if pairs:
                rs = [float(np.corrcoef(bv, cv)[0, 1]) for bv, cv in pairs]
                key_corrs[k] = rs

        if key_corrs:
            print(f"\n  {'Key':<30}  {'Mean r':>8}  {'Min r':>8}  {'Interpretation'}")
            print(f"  {'-'*30}  {'-------':>8}  {'------':>8}  {'-'*20}")
            for k, rs in sorted(key_corrs.items()):
                mean_r = np.mean(rs)
                min_r  = np.min(rs)
                if mean_r > 0.99:
                    note = 'negligible change'
                elif mean_r > 0.95:
                    note = 'minor change'
                elif mean_r > 0.90:
                    note = 'moderate change'
                else:
                    note = 'SUBSTANTIAL change'
                # Decode key: double-underscore separator from npz
                display_k = k.replace('__', '/')
                print(f"  {display_k:<30}  {mean_r:>8.4f}  {min_r:>8.4f}  {note}")

    # ── Section 3: Betti AUC changes ────────────────────────────────────────
    print('\n' + '=' * 70)
    print('BETTI AUC CHANGES  (mean absolute change across subjects)')
    print('=' * 70)

    # Find all AUC columns
    auc_cols = sorted({k for r in curr_rows for k in r if '_auc_b' in k})

    if auc_cols:
        print(f"\n  {'Feature':<30}  {'Mean |Δ|':>10}  {'Max |Δ|':>10}  {'ADHD mean Δ':>12}  {'TDC mean Δ':>12}")
        print(f"  {'-'*30}  {'--------':>10}  {'-------':>10}  {'-----------':>12}  {'----------':>12}")

        for col in auc_cols:
            deltas_adhd, deltas_tdc = [], []
            for sid in common:
                bval = base_by_sid[sid].get(col, '')
                cval = curr_by_sid[sid].get(col, '')
                try:
                    delta = float(cval) - float(bval)
                    if gmap.get(sid) == 'ADHD':
                        deltas_adhd.append(delta)
                    else:
                        deltas_tdc.append(delta)
                except (TypeError, ValueError):
                    pass

            all_deltas = deltas_adhd + deltas_tdc
            if not all_deltas:
                continue
            mean_abs = np.mean(np.abs(all_deltas))
            max_abs  = np.max(np.abs(all_deltas))
            m_adhd   = np.mean(deltas_adhd) if deltas_adhd else float('nan')
            m_tdc    = np.mean(deltas_tdc)  if deltas_tdc  else float('nan')
            print(f"  {col:<30}  {mean_abs:>10.4f}  {max_abs:>10.4f}  "
                  f"{m_adhd:>+12.4f}  {m_tdc:>+12.4f}")

    # ── Section 4: Effect size shift ─────────────────────────────────────────
    print('\n' + '=' * 70)
    print("EFFECT SIZE SHIFT  (Cohen's d: ADHD vs TDC, baseline → current)")
    print("Δd > 0.10 means the group difference meaningfully changed.")
    print('=' * 70)

    if auc_cols:
        print(f"\n  {'Feature':<30}  {'d baseline':>12}  {'d current':>12}  {'Δd':>8}  {'Direction'}")
        print(f"  {'-'*30}  {'----------':>12}  {'---------':>12}  {'--':>8}  {'-'*15}")

        for col in auc_cols:
            base_adhd, base_tdc, curr_adhd, curr_tdc = [], [], [], []
            for sid in common:
                grp  = gmap.get(sid, '')
                bval = base_by_sid[sid].get(col, '')
                cval = curr_by_sid[sid].get(col, '')
                try:
                    bv = float(bval)
                    cv = float(cval)
                except (TypeError, ValueError):
                    continue
                if grp == 'ADHD':
                    base_adhd.append(bv)
                    curr_adhd.append(cv)
                else:
                    base_tdc.append(bv)
                    curr_tdc.append(cv)

            d_base = _cohens_d(base_adhd, base_tdc)
            d_curr = _cohens_d(curr_adhd, curr_tdc)
            if np.isnan(d_base) or np.isnan(d_curr):
                continue
            delta_d = d_curr - d_base
            if abs(delta_d) < 0.05:
                direction = 'stable'
            elif delta_d > 0:
                direction = 'ADHD-TDC gap LARGER'
            else:
                direction = 'ADHD-TDC gap smaller'
            print(f"  {col:<30}  {d_base:>+12.4f}  {d_curr:>+12.4f}  "
                  f"{delta_d:>+8.4f}  {direction}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f"Comparison: '{baseline_name}' (baseline)  vs  current results")
    print(f"Subjects compared: {len(common)}  "
          f"(ADHD={sum(1 for s in common if gmap.get(s)=='ADHD')}, "
          f"TDC={sum(1 for s in common if gmap.get(s)=='TDC')})")
    print('=' * 70)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Snapshot and compare preprocessing pipeline runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Save baseline before editing code:
  python pipeline/compare_runs.py --snapshot baseline

  # After editing + rerunning test subjects:
  python pipeline/compare_runs.py --compare baseline

  # Snapshot / compare only the 10-subject test set:
  python pipeline/compare_runs.py --snapshot baseline --test-set
  python pipeline/compare_runs.py --compare baseline --test-set

  # Snapshot / compare specific subjects:
  python pipeline/compare_runs.py --snapshot baseline --subjects v10p v41p
        """,
    )
    parser.add_argument('--snapshot', metavar='NAME',
                        help='Save a snapshot of current results under NAME')
    parser.add_argument('--compare',  metavar='NAME',
                        help='Compare current results against snapshot NAME')
    parser.add_argument('--test-set', action='store_true',
                        help=f'Restrict to default 10-subject test set: {DEFAULT_TEST_SUBJECTS}')
    parser.add_argument('--subjects', nargs='+', metavar='SUBJECT_ID',
                        help='Restrict to specific subject IDs')
    parser.add_argument('--list', action='store_true',
                        help='List all available snapshots')
    args = parser.parse_args()

    if args.list:
        if not os.path.exists(SNAPSHOTS_DIR):
            print("No snapshots yet.")
        else:
            snaps = [d for d in os.listdir(SNAPSHOTS_DIR)
                     if os.path.isdir(os.path.join(SNAPSHOTS_DIR, d))]
            if not snaps:
                print("No snapshots yet.")
            else:
                for s in sorted(snaps):
                    meta = os.path.join(SNAPSHOTS_DIR, s, 'meta.txt')
                    if os.path.exists(meta):
                        print(open(meta).read().strip())
                    else:
                        print(s)
                    print()
        sys.exit(0)

    subjects_filter = None
    if args.test_set:
        subjects_filter = set(DEFAULT_TEST_SUBJECTS)
    if args.subjects:
        subjects_filter = set(args.subjects)

    if args.snapshot:
        snapshot(args.snapshot, subjects_filter=subjects_filter)
    elif args.compare:
        compare(args.compare, subjects_filter=subjects_filter)
    else:
        parser.print_help()
