"""Download BabySLM's spoken minimal pairs (https://arxiv.org/abs/2306.01506) and list them.

data/eval/babyslm/<task>/<split>/
    wavs/<filename>.wav
    gold.csv   one row per audio file, as distributed
    pairs.csv  one row per minimal pair and voice: its correct and wrong files side by side
    pairs-sample.csv  a fixed 10% of the pairs (of each type, for syntactic), every voice kept,
                      for the fast evaluation

Run it with: python -m scripts.eval.prepare_eval --models --tasks babyslm-lexical babyslm-syntactic
"""

import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from speechlm.zero_shot.config import DATA_DIR

URL = "https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/{task}/{split}.zip"
# per task: what the two files of a pair share, and what differs between them
SHARED = {"lexical": ["word", "frequency", "length"], "syntactic": ["type", "subtype"]}
DIFFERS = {"lexical": ["phones"], "syntactic": ["transcription"]}
SAMPLE_FRACTION = 0.1
SEED = 0


def task_dir(task, split="dev"):
    return DATA_DIR / "babyslm" / task / split


def prepare(task, split="dev"):
    """Download the task's audio and write its pairs.csv and pairs-sample.csv. What exists is
    skipped."""
    if not (task_dir(task, split) / "gold.csv").exists():
        download(task, split)
    if not (task_dir(task, split) / "pairs.csv").exists():
        make_pairs(task, split)
    if not (task_dir(task, split) / "pairs-sample.csv").exists():
        make_sample(task, split)


def download(task, split="dev"):
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
