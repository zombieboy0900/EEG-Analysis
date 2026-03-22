"""
status.py — progress dashboard for the ADHD EEG TDA pipeline.

Reads per-subject cache folders and run.log files to show completion state
without running any analysis.

Usage:
    python pipeline/status.py
    python pipeline/status.py --verbose    # show last log line for each subject
"""

import os
import sys
import csv
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

CHECK  = '\u2713'   # ✓
CROSS  = '\u2717'   # ✗

CACHE_FILES = {
    'Preproc':   ('epochs_clean.fif', 'ch_names.pkl'),
    'Conn':      ('conn_matrices.npz', 'dist_matrices.npz'),
    'TDA':       ('tda_results.pkl',),
    'Graph':     ('graph_results.pkl',),
    'Features':  ('betti_features.pkl',),
}


def _check(cache_dir, *files):
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in files)


def _read_status(out_dir):
    """Read last STATUS line from run.log. Returns (status_str, last_line)."""
    log_path = os.path.join(out_dir, 'run.log')
    if not os.path.exists(log_path):
        return 'pending', ''
    try:
        lines = Path(log_path).read_text(encoding='utf-8', errors='replace').splitlines()
        # Find last STATUS line
        for line in reversed(lines):
            if 'STATUS:' in line:
                if 'complete' in line:
                    return 'complete', line.strip()
                elif 'failed' in line:
                    # Extract error message after 'failed — '
                    msg = line.split('failed —', 1)[-1].strip() if 'failed —' in line else 'failed'
                    return 'failed', msg[:60]
        # Log exists but no STATUS line — process is running or died
        last = lines[-1].strip() if lines else ''
        return 'running', last[:60]
    except Exception:
        return 'unknown', ''


def _load_subjects():
    if not os.path.exists(config.METADATA_CSV):
        print(f"ERROR: metadata.csv not found at {config.METADATA_CSV}")
        sys.exit(1)
    rows = []
    with open(config.METADATA_CSV, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'subject_id': row['subject_id'].strip(),
                'group':      row.get('group', '?').strip(),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Show pipeline progress for all subjects.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show last log line for each subject')
    args = parser.parse_args()

    subjects = _load_subjects()

    step_names = list(CACHE_FILES.keys())   # ordered

    # ── Header ────────────────────────────────────────────────────────────────
    col_w   = 8
    id_w    = max(12, max(len(s['subject_id']) for s in subjects) + 1)
    grp_w   = 6
    steps_w = col_w * len(step_names)

    sep = '-' * (id_w + grp_w + steps_w + 14)
    header = (f"{'Subject':<{id_w}} {'Group':<{grp_w}}"
              + ''.join(f"{n:^{col_w}}" for n in step_names)
              + f"  {'Status'}")
    print()
    print(header)
    print(sep)

    # ── Per-subject rows ──────────────────────────────────────────────────────
    counts = {'complete': 0, 'failed': 0, 'running': 0, 'pending': 0}

    for s in subjects:
        sid      = s['subject_id']
        group    = s['group']
        out_dir  = os.path.join(config.RESULTS_ROOT, 'subjects', sid)
        cache_dir = os.path.join(out_dir, '.cache')

        step_marks = []
        for step, files in CACHE_FILES.items():
            ok = _check(cache_dir, *files)
            step_marks.append(f"{CHECK:^{col_w}}" if ok else f"{CROSS:^{col_w}}")

        status, detail = _read_status(out_dir)
        counts[status] = counts.get(status, 0) + 1

        status_display = {
            'complete': 'complete',
            'failed':   f'FAILED — {detail}' if detail else 'FAILED',
            'running':  'running...',
            'pending':  'pending',
            'unknown':  'unknown',
        }.get(status, status)

        row = (f"{sid:<{id_w}} {group:<{grp_w}}"
               + ''.join(step_marks)
               + f"  {status_display}")
        print(row)

        if args.verbose and detail and status not in ('complete', 'pending'):
            print(f"  {'':>{id_w + grp_w}}  {detail}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(sep)
    total = len(subjects)
    print(
        f"Total: {total}  |  "
        f"Complete: {counts['complete']}  |  "
        f"Running: {counts.get('running', 0)}  |  "
        f"Failed: {counts['failed']}  |  "
        f"Pending: {counts['pending']}"
    )
    print()

    # Highlight failures
    failed = [s['subject_id'] for s in subjects
              if _read_status(os.path.join(config.RESULTS_ROOT, 'subjects', s['subject_id']))[0] == 'failed']
    if failed:
        print(f"Failed subjects — check results/subjects/{{id}}/run.log:")
        for sid in failed:
            print(f"  {sid}")
        print()


if __name__ == '__main__':
    main()
