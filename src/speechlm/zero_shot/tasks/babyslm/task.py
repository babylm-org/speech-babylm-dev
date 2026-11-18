"""BabySLM's spoken minimal pairs (https://arxiv.org/abs/2306.01506), converted to a BabyLM
evaluation format.

data/eval/babyslm/<task>/<split>/
    wavs/<filename>.wav
    gold.csv   one row per audio file, as distributed
    pairs.csv  one row per minimal pair and voice: its correct and wrong files side by side
    pairs-sample.csv  a fixed 10% of the pairs (of each type, for syntactic), every voice kept,
                      for the fast evaluation

Download: python -m scripts.eval.prepare_eval --models --tasks babyslm-lexical babyslm-syntactic
"""

import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from speechlm.zero_shot.config import DATA_DIR
from speechlm.zero_shot.models.base import SpeechLM
from speechlm.zero_shot.tasks.babyslm.evaluate import score_pairs
from speechlm.zero_shot.tasks.base import EvalTask

URL = "https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/{task}/{split}.zip"
SHARED = {"lexical": ["word", "frequency", "length"], "syntactic": ["type", "subtype"]}
DIFFERS = {"lexical": ["phones"], "syntactic": ["transcription"]}
SAMPLE_FRACTION = 0.1
SEED = 0


class BabySLMEval(EvalTask):
    """Needs a speech language model: one with get_log_probs."""

    def __init__(self, task, split="dev"):
        self.task, self.split = task, split
        self.name = f"babyslm-{task}" + ("" if split == "dev" else f"-{split}")

    def setup(self):
        """Download the task's audio and write its pairs.csv and pairs-sample.csv. What exists
        is skipped."""
        out_dir = task_dir(self.task, self.split)
        if not (out_dir / "gold.csv").exists():
            download_audio(self.task, self.split)
        if not (out_dir / "pairs.csv").exists():
            make_pairs(self.task, self.split)

        self.make_fast()

    def make_fast(self):
        out_dir = task_dir(self.task, self.split)
        if not (out_dir / "pairs-sample.csv").exists():
            make_sample(self.task, self.split)

    def evaluate(self, model, batch_size=16, fast=False, **_):
        """fast: score pairs-sample.csv, a fixed 10% of the pairs."""
        if type(model).get_log_probs is SpeechLM.get_log_probs:
            print(f"Skipping {model.name} on {self.name}: not a language model")
            return

        pairs_file = task_dir(self.task, self.split) / ("pairs-sample.csv" if fast else "pairs.csv")
        out_dir = self.results_dir(model, fast)
        if (out_dir / "summary.json").exists():
            print(f"Skipping {model.name} on {self.name}: results in {out_dir}")
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
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(out_dir / "pairs.csv", index=False)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{model.name} on {self.name}: {summary['accuracy']:.4f} -> {out_dir}")


def task_dir(task, split="dev"):
    return DATA_DIR / "babyslm" / task / split


def download_audio(task, split="dev"):
    out_dir = task_dir(task, split)
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    url = URL.format(task=task, split=split)
    print(f"==> BabySLM {task} {split}: {url}")
    # extracted next to out_dir, so the files are moved rather than copied
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        archive = f"{tmp}/{split}.zip"
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        # the archive's folder is named <split> or <split>_16
        (gold,) = Path(tmp).rglob("gold.csv")
        for wav in gold.parent.glob("*.wav"):
            wav.rename(out_dir / "wavs" / wav.name)
        gold.rename(out_dir / "gold.csv")


def make_pairs(task, split="dev"):
    gold = pd.read_csv(task_dir(task, split) / "gold.csv")
    differs = ["filename", *DIFFERS[task]]
    correct = gold[gold["correct"] == 1][["id", "voice", *SHARED[task], *differs]]
    wrong = gold[gold["correct"] == 0][["id", "voice", *differs]]
    pairs = correct.merge(
        wrong, on=["id", "voice"], suffixes=("_correct", "_wrong"), validate="1:1"
    )
    if 2 * len(pairs) != len(gold):
        raise ValueError(f"{task}: not every file is in exactly one pair")
    columns = [f"{c}_{side}" for c in differs for side in ("correct", "wrong")]
    pairs = pairs[["id", "voice", *SHARED[task], *columns]].sort_values(["id", "voice"])
    pairs.to_csv(task_dir(task, split) / "pairs.csv", index=False)
    print(f"    {len(pairs)} pairs -> {task_dir(task, split) / 'pairs.csv'}")


def make_sample(task, split="dev"):
    pairs = pd.read_csv(task_dir(task, split) / "pairs.csv")
    ids = pairs.drop_duplicates("id")
    if task == "syntactic":  # so every type keeps its share
        ids = ids.groupby("type")
    ids = ids.sample(frac=SAMPLE_FRACTION, random_state=SEED)["id"]
    sample = pairs[pairs["id"].isin(ids)]
    sample.to_csv(task_dir(task, split) / "pairs-sample.csv", index=False)
    print(f"    {len(sample)} pairs -> {task_dir(task, split) / 'pairs-sample.csv'}")
