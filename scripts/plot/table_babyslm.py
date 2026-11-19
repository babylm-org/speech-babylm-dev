"""BabySLM accuracy of every model with results, as a table: one row per model, one column per
task (lexical, syntactic). Chance is 50%.

Reads results/<model>/babyslm-<task>/[fast/]summary.json from run_eval, prints the table and
saves it to results/figures/[fast/]babyslm.txt, next to the ABX plots.

Usage:
    python -m scripts.plot.table_babyslm
    python -m scripts.plot.table_babyslm --fast
"""

import argparse
import json
import re

from tabulate import tabulate

from speechlm.zero_shot.config import RESULT_DIR
from speechlm.zero_shot.tasks.summary import TABLE_FORMAT


def load_accuracies(fast=False):
    """Return {model: {task: accuracy}}, e.g. {"slam": {"lexical": 0.71, "syntactic": 0.62}}."""
    accuracies = {}
    pattern = "*/babyslm-*/fast/summary.json" if fast else "*/babyslm-*/summary.json"
    for path in RESULT_DIR.glob(pattern):
        task_dir = path.parent.parent if fast else path.parent
        task = task_dir.name.removeprefix("babyslm-")
        model = task_dir.parent.name
        accuracies.setdefault(model, {})[task] = json.loads(path.read_text())["accuracy"]
    return accuracies


def natural_key(name):
    """Sort k128 before k1024: numbers by value."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Tabulate the results of the fast evaluation (run_eval --fast)",
    )
    args = parser.parse_args()

    accuracies = load_accuracies(args.fast)
    if not accuracies:
        raise SystemExit(f"No BabySLM results in {RESULT_DIR}")
    tasks = sorted({task for by_task in accuracies.values() for task in by_task})
    rows = [
        [model, *(f"{by_task[t]:.1%}" if t in by_task else "" for t in tasks)]
        for model, by_task in sorted(accuracies.items(), key=lambda item: natural_key(item[0]))
    ]
    table = tabulate(
        rows,
        headers=["model", *tasks],
        tablefmt=TABLE_FORMAT,
        colalign=("left",) + ("right",) * len(tasks),
    )
    text = f"BabySLM accuracy{' (fast)' if args.fast else ''}, chance 50%\n{table}\n"
    print(text)
    out = RESULT_DIR / "figures" / ("fast" if args.fast else "") / "babyslm.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Saved {out}")
