"""A task's results summary: summary.json with its metrics, and summary.txt with them as tables."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate

TABLE_FORMAT = "rounded_outline"
# a column is broken down in a table when it has between these many values
MIN_LEVELS = 2
MAX_LEVELS = 25


def accuracy_by(results: pd.DataFrame, column: str) -> pd.DataFrame:
    """Items, correct items and accuracy for each value of column; results has a 0/1 "correct"."""
    return results.groupby(column)["correct"].agg(items="size", correct="sum", accuracy="mean")


def breakdown_columns(results: pd.DataFrame, exclude) -> list[str]:
    """The columns of results worth a table: those with a few distinct values."""
    return [
        c
        for c in results.columns
        if c not in exclude and MIN_LEVELS <= results[c].nunique() <= MAX_LEVELS
    ]


def write_summary(out_dir: Path, metrics: dict, tables: dict[str, pd.DataFrame]):
    """Save metrics to out_dir/summary.json, and summary.txt: metrics' single values, then the
    tables (title: table). The text is also printed.

    Fractions (floats) are shown as percentages in the text.
    """
    metrics = {**metrics, "date": datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2, default=float))

    header = [(name, show(value)) for name, value in metrics.items() if not isinstance(value, dict)]
    blocks = [rule("EVALUATION SUMMARY"), tabulate(header, tablefmt=TABLE_FORMAT)]
    for title, table in tables.items():
        shown = table.map(show).reset_index()
        blocks += [
            rule(title.upper()),
            tabulate(
                shown,
                headers="keys",
                tablefmt=TABLE_FORMAT,
                showindex=False,
                colalign=("left",) + ("right",) * (shown.shape[1] - 1),
            ),
        ]
    text = "\n".join(blocks) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def show(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "" if pd.isna(value) else f"{value:.1%}"
    if isinstance(value, int):
        return f"{value:,}"
    return value


def rule(title, width=72):
    return f"\n{title}\n{'─' * width}"
