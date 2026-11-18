"""Download everything the evaluations need: model weights, and each task's data.

Safe to re-run: what is already there is skipped.

Usage:
    python -m scripts.eval.prepare_eval
    python -m scripts.eval.prepare_eval --models facebook/wav2vec2-base cpc --tasks stress
    python -m scripts.eval.prepare_eval --models slprl/slam --tasks babyslm-lexical babyslm-syntactic
    python -m scripts.eval.prepare_eval --models --tasks triphone-dev-clean  # data only
"""

import argparse

from dotenv import load_dotenv

from speechlm.zero_shot.config import DEFAULT_MODELS, DEFAULT_TASKS, MODEL_DIR
from speechlm.zero_shot.model_registry import get_model
from speechlm.zero_shot.task_registry import TASKS, get_task

load_dotenv()  # HF_TOKEN


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Models as accepted by src/speechlm/zero_shot/model_registry.py; none to skip",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=DEFAULT_TASKS,
        choices=list(TASKS),
        help="Tasks of src/speechlm/zero_shot/task_registry.py to fetch the data of; none to skip",
    )
    args = parser.parse_args()

    for model in args.models:
        print(f"==> Model: {model}", flush=True)
        get_model(model, MODEL_DIR)  # loading fetches what the model is missing
    for task in args.tasks:
        get_task(task).setup()
    print("Done.")


if __name__ == "__main__":
    main()
