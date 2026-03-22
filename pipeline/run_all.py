"""
run_all.py — parallel batch runner for the ADHD EEG TDA pipeline.

Processes all subjects in data/metadata.csv in parallel using joblib (loky backend).
Each subject runs in its own process — MNE-safe, fully isolated.

Usage:
    python pipeline/run_all.py                        # all subjects, n_jobs from config
    python pipeline/run_all.py --n-jobs 4             # override parallelism
    python pipeline/run_all.py --force                # ignore all caches
    python pipeline/run_all.py --subjects v10p v11p   # specific subjects only
    python pipeline/run_all.py --figures              # generate per-subject figures
    python pipeline/run_all.py --group                # run group analysis after batch
"""

import os
import sys
import argparse
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from pipeline_subject import run_subject


def load_subject_ids(subjects_filter=None):
    """Read subject IDs from metadata.csv. Returns list of (subject_id, mat_path)."""
    if not os.path.exists(config.METADATA_CSV):
        print(f"ERROR: metadata.csv not found at {config.METADATA_CSV}")
        sys.exit(1)

    entries = []
    with open(config.METADATA_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row['subject_id'].strip()
            if subjects_filter and sid not in subjects_filter:
                continue
            mat_path = os.path.join(config.RAW_DIR, f'{sid}.mat')
            if not os.path.exists(mat_path):
                print(f"  WARNING: {mat_path} not found — skipping {sid}")
                continue
            entries.append((sid, mat_path))

    return entries


def main():
    parser = argparse.ArgumentParser(
        description='Run TDA pipeline for all subjects in parallel.')
    parser.add_argument('--n-jobs',   type=int, default=config.N_JOBS,
                        help=f'Number of parallel workers (default: {config.N_JOBS})')
    parser.add_argument('--force',    action='store_true',
                        help='Ignore all caches and recompute from scratch')
    parser.add_argument('--figures',  action='store_true',
                        help='Generate per-subject visualizations (slow)')
    parser.add_argument('--subjects', nargs='+', metavar='SUBJECT_ID',
                        help='Process only these subject IDs (e.g. --subjects v10p v11p)')
    parser.add_argument('--group',    action='store_true',
                        help='Run group analysis after batch completes')
    args = parser.parse_args()

    entries = load_subject_ids(subjects_filter=set(args.subjects) if args.subjects else None)

    if not entries:
        print("No subjects to process.")
        sys.exit(0)

    print(f"\nProcessing {len(entries)} subject(s) with n_jobs={args.n_jobs}")
    print(f"  force={args.force}  figures={args.figures}")
    print()

    # ── Serial fallback for n_jobs=1 (easier debugging) ──────────────────────
    if args.n_jobs == 1:
        results = []
        for sid, mat_path in entries:
            r = run_subject(mat_path, force=args.force, figures=args.figures)
            results.append((sid, r))
    else:
        # ── Parallel via joblib loky (separate processes — MNE-safe) ─────────
        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("ERROR: joblib not installed. Run: pip install joblib")
            sys.exit(1)

        mat_paths = [mat_path for _, mat_path in entries]
        sids      = [sid      for sid, _       in entries]

        raw_results = Parallel(
            n_jobs=args.n_jobs,
            backend='loky',
            verbose=10,
        )(
            delayed(run_subject)(mat_path, force=args.force, figures=args.figures)
            for mat_path in mat_paths
        )

        results = list(zip(sids, raw_results))

    # ── Summary ───────────────────────────────────────────────────────────────
    succeeded = [sid for sid, r in results if r is not None]
    failed    = [sid for sid, r in results if r is None]

    print(f"\n{'='*50}")
    print(f"Batch complete: {len(succeeded)}/{len(results)} succeeded")
    if failed:
        print(f"Failed subjects ({len(failed)}):")
        for sid in failed:
            print(f"  {sid}  — check results/subjects/{sid}/run.log")
    print(f"{'='*50}\n")

    # ── Optional group analysis ───────────────────────────────────────────────
    if args.group and succeeded:
        print("Running group analysis...")
        try:
            from pipeline_group import run_group
            run_group()
        except ImportError:
            print("pipeline_group.py not yet implemented.")

    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()
