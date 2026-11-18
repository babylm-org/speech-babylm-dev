"""BabySLM's spoken minimal pairs (https://arxiv.org/abs/2306.01506), dev split.

Data (see EvalTask; built by scripts/eval/build_eval_data.py):
    babyslm/<task>.csv  one row per minimal pair and voice: its correct and wrong files side by
                        side (filename_correct, filename_wrong)
    babyslm/<task>.zip  the audio, <filename>.wav

The fast subset is 10% of the pairs (of each type, for syntactic), every voice kept.

Results in results/<model>/<task>/[fast/]:
    pairs.csv  one row per pair and voice, with both log probabilities and its score (1 or 0)
    summary.json, summary.txt  the accuracy, overall and by voice, type, ...
"""

import pandas as pd

from speechlm.zero_shot.models.base import SpeechLM
from speechlm.zero_shot.tasks.babyslm.evaluate import score_pairs
from speechlm.zero_shot.tasks.base import EvalTask
from speechlm.zero_shot.tasks.summary import accuracy_by, breakdown_columns, write_summary


class BabySLMEval(EvalTask):
    """Needs a speech language model: one with get_log_probs."""

    family = "babyslm"

    def __init__(self, task):
        self.subset = task
        self.name = f"babyslm-{task}"

    def evaluate(self, model, batch_size=16, fast=False, **_):
        """fast: score the fast subset, 10% of the pairs."""
        if type(model).get_log_probs is SpeechLM.get_log_probs:
            print(f"Skipping {model.name} on {self.name}: not a language model")
            return

        out_dir = self.results_dir(model, fast)
        if (out_dir / "summary.json").exists():
            print(f"Skipping {model.name} on {self.name}: results in {out_dir}")
            return

        pairs = score_pairs(
            model,
            pd.read_csv(self.data_dir(fast) / f"{self.subset}.csv"),
            self.audio_dir(fast),
            batch_size,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(out_dir / "pairs.csv", index=False)

        results = pairs.assign(correct=pairs["score"])
        # the pair's own columns (files, log probabilities...) are not conditions
        sides = [c for c in results if c.endswith(("_correct", "_wrong"))]
        tables = {
            f"accuracy by {column}": accuracy_by(results, column)
            for column in breakdown_columns(results, exclude=["id", "score", "correct", *sides])
        }
        metrics = {"model": model.name, "task": self.name, "fast": fast, "pairs": len(pairs)}
        metrics["chance"] = 0.5
        metrics["accuracy"] = float(results["correct"].mean())
        for title, table in tables.items():
            metrics[title] = {str(k): float(a) for k, a in table["accuracy"].items()}
        write_summary(out_dir, metrics, tables)
