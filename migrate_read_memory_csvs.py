"""
Migrate old-style read_memory CSV files to the new two-file format.

Old format (single file):
    read_memory_cycles{N}_eps{eps}.csv
    columns: perturbation_size, n_changed_weights, delta_weight_0, delta_weight_1, ...

New format (two files):
    read_memory_cycles{N}_eps{eps}.csv  →  perturbation_size, n_changed_weights
    read_memory_cycles{N}_deltas.csv    →  perturbation_size, delta_weight_0, delta_weight_1, ...

Usage:
    python migrate_read_memory_csvs.py /path/to/directory
"""

import argparse
import csv
import os
import re
import sys


OLD_PATTERN = re.compile(r"^read_memory_cycles(\d+)_eps(.+)\.csv$")


def is_old_style(fieldnames: list[str]) -> bool:
    return "n_changed_weights" in fieldnames and any(
        f.startswith("delta_weight_") for f in fieldnames
    )


def migrate_file(csv_path: str, dry_run: bool) -> None:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not is_old_style(fieldnames):
            print(f"  SKIP (already new format or unrecognised): {os.path.basename(csv_path)}")
            return
        rows = list(reader)

    delta_cols = [c for c in fieldnames if c.startswith("delta_weight_")]
    base = os.path.dirname(csv_path)
    fname = os.path.basename(csv_path)
    m = OLD_PATTERN.match(fname)
    cycle_str = m.group(1)

    summary_path = csv_path  # overwrite in-place
    delta_path = os.path.join(base, f"read_memory_cycles{cycle_str}_deltas.csv")

    if dry_run:
        print(f"  [dry-run] would write summary → {os.path.basename(summary_path)}")
        print(f"  [dry-run] would write deltas  → {os.path.basename(delta_path)}")
        return

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["perturbation_size", "n_changed_weights"])
        for row in rows:
            writer.writerow([row["perturbation_size"], row["n_changed_weights"]])

    with open(delta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["perturbation_size"] + delta_cols)
        for row in rows:
            writer.writerow([row["perturbation_size"]] + [row[c] for c in delta_cols])

    print(f"  summary → {os.path.basename(summary_path)}")
    print(f"  deltas  → {os.path.basename(delta_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate read_memory CSVs to new two-file format.")
    parser.add_argument("--directory", type=str, default=None, required=True, help="Directory containing old-style CSV files.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing any files.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        sys.exit(f"Error: '{args.directory}' is not a directory.")

    candidates = sorted(
        f for f in os.listdir(args.directory) if OLD_PATTERN.match(f)
    )

    if not candidates:
        print("No matching files found.")
        return

    print(f"Found {len(candidates)} candidate file(s) in '{args.directory}':\n")
    for fname in candidates:
        print(f"Processing: {fname}")
        migrate_file(os.path.join(args.directory, fname), dry_run=args.dry_run)
        print()


if __name__ == "__main__":
    main()
