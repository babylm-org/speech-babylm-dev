"""BabySLM's spoken lexical (sWUGGY-like) and syntactic (sBLIMP-like) minimal pairs.

Results: results/<model>/babyslm/<pairs file>.csv, the pairs with their log probabilities and
scores, and <pairs file>.json, the accuracy (also by type of syntactic pair), where the pairs
file is <task>_<split>, or <task>_<split>-sample for the fast evaluation.
"""

import json

import pandas as pd

from speechlm.zero_shot.config import results_dir
from speechlm.zero_shot.models.base import SpeechLM
from speechlm.zero_shot.tasks.babyslm.evaluate import score_pairs
from speechlm.zero_shot.tasks.babyslm.prepare import prepare, task_dir
from speechlm.zero_shot.tasks.base import EvalTask


class BabySLMEval(EvalTask):
    """Needs a speech language model: one with get_log_probs."""

    def __init__(self, task, split="dev"):
        self.task, self.split = task, split
        self.name = f"babyslm-{task}" + ("" if split == "dev" else f"-{split}")

    def download(self):
        prepare(self.task, self.split)

    def evaluate(self, model, batch_size=16, fast=False, **_):
        """fast: score pairs-sample.csv, a fixed 10% of the pairs."""
        if type(model).get_log_probs is SpeechLM.get_log_probs:
            print(f"Skipping {model.name} on {self.name}: not a language model")
            return
        pairs_file = task_dir(self.task, self.split) / (
            "pairs-sample.csv" if fast else "pairs.csv"
        )
        name = f"{self.task}_{self.split}" + ("-sample" if fast else "")
        stem = results_dir(model.name, "babyslm") / name
        if stem.with_suffix(".json").exists():
            print(f"Skipping {model.name} on {self.name}: results in {stem}.json")
            return

        pairs = score_pairs(
            model,
            pd.read_csv(pairs_file),
            task_dir(self.task, self.split) / "wavs",
            batch_size,
        )

        summary = {"accuracy": pairs["score"].mean(), "pairs": len(pairs)}
        if "type" in pairs:
            summary["by_type"] = pairs.groupby("type")["score"].mean().to_dict()
        stem.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(stem.with_suffix(".csv"), index=False)
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        print(f"{model.name} on {self.name}: {summary['accuracy']:.4f} -> {stem}.json")
