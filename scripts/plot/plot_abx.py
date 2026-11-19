"""Plot ABX accuracy per layer in one figure: one row per task, one column per model, as in
internal_tools/tutorials/2_activation_analyses.ipynb.

Reads results/{full,fast}/<model>/<task>/abx_*.csv from run_eval: the line is the mean accuracy
over contrasts (= the overall ABX accuracy), the band its 95% CI. Saves
results/figures/{full,fast}/abx.png.

Usage:
    python -m scripts.plot.plot_abx
    python -m scripts.plot.plot_abx --fast
    python -m scripts.plot.plot_abx --models wav2vec2-base hubert-base-ls960 --tasks triphone-dev-clean stress
"""

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from speechlm.zero_shot.config import DEFAULT_MODELS, DEFAULT_TASKS, RESULT_DIR

CONDITION_STYLES = {
    "within-speaker": dict(color=sns.color_palette("Paired")[3]),
    "across-speaker": dict(color=sns.color_palette("Paired")[2]),
    "random": dict(color="grey", dashes=(1, 2)),
}
# a model's panel title is wrapped at hyphens to lines of at most this many characters
TITLE_WIDTH = 14


def find_results(fast=False):
    """Return {(task, model name): DataFrame of all its abx_*.csv results}."""
    files = {}
    for csv in (RESULT_DIR / ("fast" if fast else "full")).glob("*/*/abx_*.csv"):
        files.setdefault((csv.parent.name, csv.parent.parent.name), []).append(csv)
    return {key: pd.concat(pd.read_csv(f) for f in sorted(csvs)) for key, csvs in files.items()}


def model_order(name):
    """DEFAULT_MODELS first, in their order, then the rest with numbers by value (k128 < k1024)."""
    order = [Path(m).name for m in DEFAULT_MODELS]
    parts = [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]
    return (order.index(name) if name in order else len(order), parts)


def wrap_title(name, width=TITLE_WIDTH):
    """The model name over several lines, broken after hyphens."""
    lines = [""]
    for part in re.findall(r"[^-]+-?", name):
        if lines[-1] and len(lines[-1]) + len(part) > width:
            lines.append("")
        lines[-1] += part
    return "\n".join(lines)


def plot_ABX_results(results: dict, tasks, models, title="", filepath=None):
    """One row per task, one column per model, one line per condition."""
    sns.set_style("whitegrid")
    mpl.rcParams["axes.linewidth"] = 2
    mpl.rcParams["axes.edgecolor"] = "0.9"

    # the layers of each model, in depth order, from any task it has results for
    layers = {}
    for (_, model), df in results.items():
        if model in models:
            depths = df.drop_duplicates("layer").sort_values("depth")
            layers[model] = dict(zip(depths["layer"], depths["depth"]))
    # panel width proportional to the number of layers, e.g. a narrow one for CPC
    n_layers = [len(layers[m]) for m in models]
    width_ratios = [max(n, 3) / max(n_layers) for n in n_layers]

    fig, axs = plt.subplots(
        len(tasks),
        len(models),
        figsize=(2.25 * sum(width_ratios), 2.4 * len(tasks)),
        width_ratios=width_ratios,
        sharey=True,
        squeeze=False,
    )
    for row, task in enumerate(tasks):
        for col, model in enumerate(models):
            ax = axs[row, col]
            df = results.get((task, model))
            if df is None:
                ax.text(
                    0.5,
                    0.5,
                    "no results",
                    ha="center",
                    va="center",
                    color="0.6",
                    transform=ax.transAxes,
                )
            else:
                for condition, style in CONDITION_STYLES.items():
                    condition_results = df[df["condition"] == condition]
                    if not condition_results.empty:
                        sns.lineplot(
                            condition_results,
                            x="depth",
                            y="accuracy",
                            ax=ax,
                            label=condition,
                            **style,
                        )
                ax.legend().remove()
            depths = layers[model]
            # every other layer's label, and always the last, so they do not overlap
            names = list(depths)
            shown = [n if i % 2 == 0 or i == len(names) - 1 else "" for i, n in enumerate(names)]
            ax.set_xticks(list(depths.values()), shown if row == len(tasks) - 1 else [])
            ax.tick_params(axis="x", labelrotation=90, labelsize=8)
            ax.set_xlim(min(depths.values()), max(depths.values()))
            ax.set_xlabel("")
            ax.set_ylim(0.4, 1)
            ax.set_ylabel(f"{task}\naccuracy" if col == 0 else "")
            if row == 0:
                ax.set_title(wrap_title(model), ha="left", x=0, fontsize=10)

    # one legend for the whole figure
    handles = {}
    for ax in axs.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    # in the order of CONDITION_STYLES, whichever panel each condition first appeared in
    handles = {c: handles[c] for c in CONDITION_STYLES if c in handles}
    # legend above the panels, title above the legend, so they never overlap
    fig.legend(
        handles.values(),
        handles.keys(),
        frameon=False,
        ncol=len(handles),
        loc="lower right",
        bbox_to_anchor=(1, 1.0),
    )
    if title:
        fig.suptitle(title.upper(), ha="left", x=0, y=1.06, fontweight="bold")
    fig.tight_layout()
    if filepath:
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(filepath, dpi=200, facecolor="white", bbox_inches="tight")
        print(f"Saved {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Plot the results of the fast evaluation (run_eval --fast)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model folders to plot, as columns in this order; all with results by default",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Tasks to plot, as rows in this order; all with results by default",
    )
    parser.add_argument("--title", default="ABX tests, trained models", help="Figure title")
    args = parser.parse_args()

    results = find_results(args.fast)
    if not results:
        raise SystemExit(f"No ABX results in {RESULT_DIR}")
    # DEFAULT_TASKS first, in their order, then the rest
    found = {task for task, _ in results}
    tasks = args.tasks or [t for t in DEFAULT_TASKS if t in found] + sorted(
        found - set(DEFAULT_TASKS)
    )
    models = args.models or sorted({model for _, model in results}, key=model_order)
    results = {k: v for k, v in results.items() if k[0] in tasks and k[1] in models}
    if missing := [m for m in models if not any(m == model for _, model in results)]:
        raise SystemExit(f"No results for {missing} on {tasks}")
    plot_ABX_results(
        results,
        tasks,
        models,
        title=args.title + (" (fast)" if args.fast else ""),
        filepath=RESULT_DIR / "figures" / ("fast" if args.fast else "full") / "abx.png",
    )
