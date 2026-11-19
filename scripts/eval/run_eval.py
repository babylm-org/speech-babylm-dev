"""Evaluate each model on each task of src/speechlm/zero_shot/task_registry.py.

Results that already exist are skipped, so a failed run can simply be restarted.
--fast is the fast evaluation: each task on its fixed sample (triphone: ~10% of the split;
prosodic: the whole set, already small; BabySLM: 10% of the pairs), with its results kept apart.
Download everything first with scripts/eval/prepare_eval.py.

Usage:
    python -m scripts.eval.run_eval
    python -m scripts.eval.run_eval --models facebook/wav2vec2-base cpc --tasks triphone-dev-clean stress
    python -m scripts.eval.run_eval --models slprl/slam --tasks babyslm-lexical babyslm-syntactic
    python -m scripts.eval.run_eval --fast
"""

import os

# less GPU memory fragmentation when each layer's features are loaded (set before importing torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import traceback

import torch

from speechlm.zero_shot.config import DEFAULT_MODELS, DEFAULT_TASKS, MODEL_DIR
from speechlm.zero_shot.model_registry import get_model
from speechlm.zero_shot.task_registry import TASKS, get_task


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models as accepted by src/speechlm/zero_shot/model_registry.py",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=list(TASKS),
        help="Tasks to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Files per forward pass",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast evaluation: each task on its fixed sample, results kept apart",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        help="ABX: score only these layers, by name (e.g. CNN T6) or position (0 the first, "
        "-1 the last), those each model has; all if not given",
    )
    parser.add_argument(
        "--keep_activations",
        action="store_true",
        help="ABX: do not delete activations afterwards",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # keep going if one model fails, and report the failures at the end
    failed = []
    for model_name_or_path in args.models:
        print(f"\n=== Loading {model_name_or_path} ===")
        try:
            model = get_model(model_name_or_path, MODEL_DIR).to(device)
        except Exception:
            traceback.print_exc()
            failed.append(f"{model_name_or_path}: loading")
            continue
        for task_name in args.tasks:
            print(f"\n=== {model_name_or_path} on {task_name} ===")
            try:
                get_task(task_name).evaluate(
                    model,
                    batch_size=args.batch_size,
                    fast=args.fast,
                    layers=args.layers,
                    keep_activations=args.keep_activations,
                )
            except Exception:
                traceback.print_exc()
                failed.append(f"{model_name_or_path} on {task_name}")
        # free the model before loading the next
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if failed:
        print("\nFailed:\n  " + "\n  ".join(failed))
        raise SystemExit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
