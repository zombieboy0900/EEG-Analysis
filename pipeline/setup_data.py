"""
setup_data.py — populate data/raw/ and metadata.csv from extracted zip folders.

Point this script at the folders you extracted from the dataset zips:
  ADHD_part1.zip, ADHD_part2.zip    → pass with --adhd
  Control_part1.zip, Control_part2.zip → pass with --tdc

The script:
  1. Finds all .mat files in the given folders (recursive)
  2. Copies them to data/raw/  (skips if already there)
  3. Appends new subjects to data/metadata.csv  (skips existing entries)
  4. Prints a summary

Usage:
    python pipeline/setup_data.py --adhd "C:/Users/Owner/Desktop/ADHD_part1" "C:/Users/Owner/Desktop/ADHD_part2"
                                  --tdc  "C:/Users/Owner/Desktop/Control_part1" "C:/Users/Owner/Desktop/Control_part2"

    # Dry run — shows what would happen without copying anything
    python pipeline/setup_data.py --adhd ... --tdc ... --dry-run
"""

import os
import sys
import csv
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def find_mat_files(folders):
    """Return sorted list of .mat file paths found in any of the given folders."""
    found = []
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            print(f"  WARNING: folder not found: {folder}")
            continue
        found.extend(sorted(folder.rglob('*.mat')))
    return found


def load_existing_metadata():
    """Return set of subject_ids already in metadata.csv."""
    existing = {}
    if not os.path.exists(config.METADATA_CSV):
        return existing
    with open(config.METADATA_CSV, newline='') as f:
        for row in csv.DictReader(f):
            sid = row['subject_id'].strip()
            existing[sid] = row.get('group', '').strip()
    return existing


def main():
    parser = argparse.ArgumentParser(
        description='Populate data/raw/ and metadata.csv from extracted zip folders.')
    parser.add_argument('--adhd', nargs='+', metavar='FOLDER',
                        help='Folder(s) containing ADHD .mat files')
    parser.add_argument('--tdc',  nargs='+', metavar='FOLDER',
                        help='Folder(s) containing TDC (control) .mat files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without copying or writing anything')
    args = parser.parse_args()

    if not args.adhd and not args.tdc:
        parser.print_help()
        sys.exit(0)

    os.makedirs(config.RAW_DIR, exist_ok=True)

    adhd_files = find_mat_files(args.adhd or [])
    tdc_files  = find_mat_files(args.tdc  or [])

    print(f"\nFound {len(adhd_files)} ADHD .mat file(s)")
    print(f"Found {len(tdc_files)}  TDC  .mat file(s)")

    existing = load_existing_metadata()

    # Collect new entries
    to_copy   = []   # (src_path, dest_path, subject_id, group)
    to_add    = []   # (subject_id, group)
    skipped   = []   # subject_ids already present
    conflicts = []   # subject_ids in both ADHD and TDC lists

    adhd_ids = {Path(f).stem for f in adhd_files}
    tdc_ids  = {Path(f).stem for f in tdc_files}
    both     = adhd_ids & tdc_ids
    if both:
        conflicts = sorted(both)
        print(f"\n  WARNING: {len(conflicts)} subject ID(s) appear in both ADHD and TDC folders:")
        for sid in conflicts:
            print(f"    {sid}")
        print("  These will be skipped — resolve manually.")

    for mat_path, group in [(f, 'ADHD') for f in adhd_files] + [(f, 'TDC') for f in tdc_files]:
        sid  = Path(mat_path).stem
        dest = Path(config.RAW_DIR) / f'{sid}.mat'

        if sid in conflicts:
            continue

        if sid in existing:
            skipped.append(sid)
            continue

        to_copy.append((mat_path, dest, sid, group))
        to_add.append((sid, group))

    # ── Summary before acting ─────────────────────────────────────────────────
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Actions:")
    print(f"  Copy to data/raw/:       {len(to_copy)} file(s)")
    print(f"  Add to metadata.csv:     {len(to_add)} subject(s)")
    print(f"  Already in metadata.csv: {len(skipped)} subject(s) (skipped)")

    if args.dry_run:
        print("\nFiles that would be copied:")
        for src, dest, sid, group in to_copy:
            print(f"  [{group:<4}] {Path(src).name} -> data/raw/{sid}.mat")
        print("\nRun without --dry-run to apply.")
        return

    if not to_copy:
        print("\nNothing to do.")
        return

    # ── Copy files ────────────────────────────────────────────────────────────
    print("\nCopying files...")
    copied = 0
    for src, dest, sid, group in to_copy:
        if dest.exists():
            print(f"  SKIP (already exists): {dest.name}")
            continue
        shutil.copy2(src, dest)
        print(f"  [{group:<4}] {dest.name}")
        copied += 1

    # ── Write metadata.csv ────────────────────────────────────────────────────
    # Preserve existing rows, append new ones, sort by subject_id
    all_entries = dict(existing)
    for sid, group in to_add:
        all_entries[sid] = group

    with open(config.METADATA_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'group'])
        for sid in sorted(all_entries.keys()):
            writer.writerow([sid, all_entries[sid]])

    print(f"\nmetadata.csv updated: {len(all_entries)} total subjects")
    print(f"  ADHD: {sum(1 for g in all_entries.values() if g == 'ADHD')}")
    print(f"  TDC:  {sum(1 for g in all_entries.values() if g == 'TDC')}")

    print(f"\nDone. {copied} file(s) copied.")
    print("\nNext steps:")
    print("  # Run all subjects:")
    print("  python pipeline/run_all.py")
    print()
    print("  # Run specific subjects:")
    print("  python pipeline/run_all.py --subjects v10p v11p v107")
    print()
    print("  # Check progress:")
    print("  python pipeline/status.py")


if __name__ == '__main__':
    main()
