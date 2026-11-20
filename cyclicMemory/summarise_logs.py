"""
summarise_logs.py
-----------------
Scans all subdirectories of RESULTS_DIR for log.csv files, reports the final
train_loss of each run, and lists runs whose final train_loss < 1e-7.

Output is written to summarise_logs_output.txt next to this script.
"""

import os
import csv
import sys

RESULTS_DIR = os.path.join(os.environ.get("CYCLIC_MEMORY_PROJECT_BASE", "/path/to/cyclicMemory"), "snellius_results")
THRESHOLD = 1e-7
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summarise_logs_output.txt")


def get_final_train_loss(log_path):
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            last_row = row
    if last_row is None:
        return None
    return float(last_row["train_loss"])


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    entries = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        subdir = os.path.join(RESULTS_DIR, name)
        log_path = os.path.join(subdir, "log.csv")
        if os.path.isdir(subdir) and os.path.isfile(log_path):
            loss = get_final_train_loss(log_path)
            if loss is not None:
                entries.append((name, subdir, loss))

    below_threshold = [full_path for _, full_path, loss in entries if loss < THRESHOLD]

    with open(OUTPUT_FILE, "w") as out:
        out.write("Final train_loss per run\n")
        out.write("=" * 50 + "\n")
        for name, _, loss in entries:
            out.write(f"{name}: {loss:.6e}\n")

        out.write("\n")
        out.write(f"Runs with final train_loss < {THRESHOLD:.0e}\n")
        out.write("=" * 50 + "\n")
        if below_threshold:
            out.write("\n".join(below_threshold) + "\n")
        else:
            out.write("(none)\n")

    print(f"Output written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()