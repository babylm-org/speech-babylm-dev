"""Plot ABX accuracy per layer, one panel per model, as in internal_tools/tutorials/2_activation_analyses.ipynb.

Reads results/<model>/<task>/[fast/]abx_*.csv from run_eval: the line is the mean accuracy over
contrasts (= the overall ABX accuracy), the band its 95% CI. Draws one figure per task,
results/figures/[fast/]abx_<task>.png, and stacks them into abx_all.png.

Usage:
    python -m scripts.plot.plot_abx
    python -m scripts.plot.plot_abx --fast
    python -m scripts.plot.plot_abx --models wav2vec2-base hubert-base-ls960 --tasks triphone-dev-clean
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from speechlm.zero_shot.config import DEFAULT_MODELS, RESULT_DIR

CONDITION_STYLES = {
    "within-speaker": dict(color=sns.color_palette("Paired")[3]),
    "across-speaker": dict(color=sns.color_palette("Paired")[2]),
    "random": dict(color="grey", dashes=(1, 2)),
}


def load_results(model_names, task="triphone-dev-clean", fast=False):
    """Return {model name: DataFrame of all its abx_*.csv results for the task}."""
    results = {}
    for model_name in model_names:
        results_dir = RESULT_DIR / model_name / task / ("fast" if fast else "")
        files = sorted(results_dir.glob("abx_*.csv"))
        if not files:
            raise FileNotFoundError(f"No abx_*.csv results in {results_dir}")
        results[model_name] = pd.concat([pd.read_csv(f) for f in files])
    return results


def plot_ABX_results(ABX_results: dict, title="", filepath=None, show=True):
    """One panel per model, one line per condition present in that model's results."""
    sns.set_style("whitegrid")
    mpl.rcParams["axes.linewidth"] = 2
    mpl.rcParams["axes.edgecolor"] = "0.9"

    model_names = list(ABX_results)
    # panel width proportional to the number of layers, e.g. a narrow one for CPC
    n_layers = [ABX_results[m]["layer"].nunique() for m in model_names]
    width_ratios = [max(n, 2) / max(n_layers) for n in n_layers]

    fig, axs = plt.subplots(
        1,
        len(model_names),
        figsize=(2.25 * sum(width_ratios), 3),
        width_ratios=width_ratios,
        sharey=True,
        squeeze=False,
    )
    for ax, model_name in zip(axs[0], model_names):
        results = ABX_results[model_name]
        for condition, style in CONDITION_STYLES.items():
            condition_results = results[results["condition"] == condition]
            if not condition_results.empty:
                sns.lineplot(
                    condition_results,
                    x="depth",
                    y="accuracy",
                    ax=ax,
                    label=condition,
                    **style,
                )
        layer_depths = dict(zip(results["layer"], results["depth"]))
        ax.set_xticks(
            list(layer_depths.values()), list(layer_depths.keys()), rotation=90
        )
        ax.set_xlim(min(layer_depths.values()), max(layer_depths.values()))
        ax.set_xlabel("")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.45, 1)
        ax.legend().remove()
        ax.set_title(model_name, ha="left", x=0)

    # one legend for the whole figure
    handles = {}
    for ax in axs[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    # legend above the panels, title above the legend, so they never overlap
    fig.legend(
        handles.values(),
        handles.keys(),
        frameon=False,
        ncol=len(handles),
        loc="lower right",
        bbox_to_anchor=(1, 0.98),
    )
    if title:
        fig.suptitle(title.upper(), ha="left", x=0, y=1.12, fontweight="bold")
    if filepath:
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(filepath, dpi=300, facecolor="white", bbox_inches="tight")
    if show:
        plt.show()


def find_results(fast=False):
    """Return {task: [model names with results for it]}, models in DEFAULT_MODELS order, then the
    rest."""
    order = [Path(m).name for m in DEFAULT_MODELS]
    found = {}
    pattern = "*/*/fast/abx_*.csv" if fast else "*/*/abx_*.csv"
    for csv in sorted(RESULT_DIR.glob(pattern)):
        task_dir = csv.parent.parent if fast else csv.parent
        model_name, task = task_dir.parent.name, task_dir.name
        found.setdefault(task, set()).add(model_name)
    rank = lambda m: (order.index(m) if m in order else len(order), m)
    return {task: sorted(models, key=rank) for task, models in sorted(found.items())}


def stack_images(paths, filepath):
    """Stack figures vertically into one image, left-aligned on white."""
    images = [plt.imread(p)[..., :3] for p in paths]  # RGB floats in [0, 1]
    width = max(image.shape[1] for image in images)
    padded = [
        np.pad(
            image, ((0, 0), (0, width - image.shape[1]), (0, 0)), constant_values=1.0
        )
        for image in images
    ]
    plt.imsave(filepath, np.concatenate(padded))
    print(f"Saved {filepath}")


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
        help="Model folders to plot; all with results by default",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Tasks to plot; all with results by default",
    )
    parser.add_argument(
        "--title",
        default="ABX tests, trained models",
        help="Figure title, followed by the task name",
    )
    args = parser.parse_args()

    figures_dir = RESULT_DIR / "figures" / ("fast" if args.fast else "")
    saved = []
    for task, model_names in find_results(args.fast).items():
        if args.tasks and task not in args.tasks:
            continue
        if args.models:
            model_names = [m for m in model_names if m in args.models]
        if not model_names:
            continue
        filepath = figures_dir / f"abx_{task}.png"
        plot_ABX_results(
            load_results(model_names, task, args.fast),
            title=f"{args.title}, {task}" + (" (fast)" if args.fast else ""),
            filepath=filepath,
            show=False,
        )
        plt.close("all")
        print(f"Saved {filepath}")
        saved.append(filepath)

    if len(saved) > 1:
        stack_images(saved, figures_dir / "abx_all.png")
