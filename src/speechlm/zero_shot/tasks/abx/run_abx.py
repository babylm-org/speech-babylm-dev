import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch

from speechlm.zero_shot.config import ACTIVATION_DIR, results_dir
from speechlm.zero_shot.tasks.abx.conditions import CONDITIONS, ABXCondition


def evaluate_abx(
    model_name: str,
    path_items: str | Path,
    conditions: list[ABXCondition],
    activations_dir: str | Path = ACTIVATION_DIR,
    layers: list[str] | None = None,
    out_dir: str | Path | None = None,
):
    """
    Evaluate saved activations on ABX conditions, loading each layer's features once for all conditions.

    Args:
        model_name (str): The model's folder in activations_dir (e.g. "wav2vec2-base").
        path_items (str | Path): The path to the items file for the ABX conditions.
        conditions (list[ABXCondition]): The conditions (see conditions.py) to score on this item file.
        activations_dir (str | Path): Where get_activations.py saved the activations.
        layers (list[str] | None): Only score these layers; all extracted layers if None.
        out_dir (str | Path | None): Results go to <out_dir>/<item file name>/abx_<condition id>.csv;
            out_dir is results/<model>/abx/ if None.
    """
    out_dir = Path(out_dir or results_dir(model_name, "abx"))
    model_dir = Path(activations_dir) / model_name
    info = json.loads((model_dir / "info.json").read_text())
    results = {condition.id: [] for condition in conditions}

    for depth, layer in enumerate(info["layers"]):
        if layers is not None and layer not in layers:
            continue
        datasets = {}  # conditions of the same class share the loaded features
        for condition in conditions:
            if type(condition) not in datasets:
                datasets[type(condition)] = condition.load_dataset(
                    path_items, model_dir / layer, frequency=info["frequency"]
                )
            pair_scores = condition.score(datasets[type(condition)])
            # one row per contrast; their mean accuracy is the overall ABX accuracy
            for pair in pair_scores:
                labels = {k: v for k, v in pair.items() if k != "score"}
                results[condition.id].append(
                    {
                        "model": model_name,
                        "condition": condition.label,
                        "layer": layer,
                        "depth": depth,
                        **labels,
                        "error_rate": pair["score"],
                        "accuracy": 1 - pair["score"],
                        "item": str(path_items),
                        "task": condition.id,
                    }
                )
            error_rate = sum(pair["score"] for pair in pair_scores) / len(pair_scores)
            print(f"{layer} {condition.id}: accuracy {1 - error_rate:.4f}")
        # free this layer's features on the GPU before loading the next
        del datasets
        gc.collect()
        torch.cuda.empty_cache()

    for condition_id, rows in results.items():
        results_path = out_dir / Path(path_items).stem / f"abx_{condition_id}.csv"
        results_path.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(rows).to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Model folder in the activations directory, e.g. wav2vec2-base",
    )
    parser.add_argument("path_items", type=str, help="Path to the items file")
    parser.add_argument(
        "conditions", type=str, nargs="+", choices=list(CONDITIONS), help="ABX conditions to run"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default=ACTIVATION_DIR,
        help="Where get_activations.py saved the activations",
    )
    args = parser.parse_args()
    evaluate_abx(
        args.model_name,
        path_items=args.path_items,
        conditions=[CONDITIONS[condition_id] for condition_id in args.conditions],
        activations_dir=args.activations_dir,
    )
