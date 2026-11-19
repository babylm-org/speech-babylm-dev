"""ABX evaluation sets: extract a model's activations (get_activations.py), score every layer
on the set's ABX conditions (conditions.py, e.g. within- and across-speaker) with run_abx.py,
then delete the activations, so only one model and set is on disk at a time.

Data (see EvalTask; built by scripts/eval/build_eval_data.py):
    triphone/<split>.item, <split>.zip: LibriSpeech with the ZeroSpeech 2021 item files
    prosodic/<set>.csv, <set>.zip: English lexical stress (HaitongSUN/prosody-abx), one word
        per file, 16 kHz mono

Results in results/{full,fast}/<model>/<task>/:
    abx_<condition id>.csv  one row per layer and contrast (e.g. phone pair)
    summary.json, summary.txt  the accuracy of each layer in each condition
"""

import gc
import shutil
from pathlib import Path

import pandas as pd
import polars as pl
import torch

from speechlm.zero_shot.config import ACTIVATION_DIR
from speechlm.zero_shot.tasks.abx.conditions import (
    PROSODIC_CONDITIONS,
    TRIPHONE_CONDITIONS,
    ABXCondition,
)
from speechlm.zero_shot.tasks.abx.get_activations import extract
from speechlm.zero_shot.tasks.abx.run_abx import evaluate_abx
from speechlm.zero_shot.tasks.base import EvalTask
from speechlm.zero_shot.tasks.summary import write_summary


class ABXEval(EvalTask):
    """An item file over a folder of audio, scored on some ABX conditions.

    file_format: the audio files' extension
    items_suffix: the item file's extension: ".item" (space-separated) or ".csv"
    """

    file_format: str
    items_suffix: str
    conditions: list[ABXCondition]

    def items_file(self, fast=False) -> Path:
        return self.data_dir(fast) / f"{self.subset}{self.items_suffix}"

    def evaluate(
        self,
        model,
        batch_size=16,
        fast=False,
        layers=None,
        keep_activations=False,
        **_,
    ):
        """
        Args:
            batch_size: files per forward pass in extraction
            fast: score the fast subset
            layers: score only these layers, those the model has: names (e.g. ["CNN", "T6"]) or
                positions in the model's layers ("0" the first, "-1" the last); all if None
            keep_activations: do not delete the activations afterwards
        """
        model_name = model.name
        path_items = self.items_file(fast)
        out_dir = self.results_dir(model, fast)
        conditions = [c for c in self.conditions if not (out_dir / f"abx_{c.id}.csv").exists()]
        if conditions:
            self.score(model, fast, conditions, out_dir, batch_size, layers, keep_activations)
        else:
            print(f"Skipping {model_name} on {self.name}: results already in {out_dir}")
        self.summarize(model, fast, path_items, out_dir)

    def score(self, model, fast, conditions, out_dir, batch_size, layers, keep_activations):
        """Extract the activations, score them on the conditions, then delete them."""
        model_name = model.name
        path_items = self.items_file(fast)
        try:
            extract(
                model,
                self.audio_dir(fast),
                file_format=self.file_format,
                out_dir=ACTIVATION_DIR,
                batch_size=batch_size,
                file_ids=set(read_items(path_items)["#file"]),
                layers=layers,
            )
            gc.collect()
            torch.cuda.empty_cache()
            evaluate_abx(
                model_name,
                path_items,
                conditions,
                out_dir,
                activations_dir=ACTIVATION_DIR,
            )
        finally:
            # also on failure, so no activations are left behind
            if not keep_activations:
                shutil.rmtree(ACTIVATION_DIR / model_name, ignore_errors=True)

    def summarize(self, model, fast, path_items, out_dir):
        """The accuracy of each layer in each condition, the mean over the contrasts."""
        results = pd.concat(pd.read_csv(out_dir / f"abx_{c.id}.csv") for c in self.conditions)
        by_layer = results.pivot_table(
            index=["depth", "layer"], columns="condition", values="accuracy", aggfunc="mean"
        ).droplevel("depth")[[c.label for c in self.conditions]]
        metrics = {"model": model.name, "task": self.name, "fast": fast, "items": str(path_items)}
        metrics["chance"] = 0.5
        for label, accuracy in by_layer.items():
            metrics[f"best {label}"] = f"{accuracy.idxmax()} ({accuracy.max():.1%})"
        metrics["accuracy"] = {
            label: {layer: float(a) for layer, a in accuracy.items()}
            for label, accuracy in by_layer.items()
        }
        write_summary(out_dir, metrics, {"accuracy by layer": by_layer})


class TriphoneEval(ABXEval):
    """ZeroSpeech 2021 triphone ABX on a LibriSpeech split. Its fast subset is 10 speakers
    (5 F, 5 M) x 25 recordings, ~10% of the split."""

    family = "triphone"
    file_format = "flac"
    items_suffix = ".item"
    conditions = TRIPHONE_CONDITIONS

    def __init__(self, split):
        self.subset = split
        self.name = f"triphone-{split}"


class ProsodicEval(ABXEval):
    """Prosodic ABX (English stress) on one of its sets. The sets are small, so the fast subset
    is the whole set."""

    family = "prosodic"
    file_format = "wav"
    items_suffix = ".csv"
    conditions = PROSODIC_CONDITIONS

    def __init__(self, set_name):
        self.subset = self.name = set_name


def read_items(path_items):
    """An item file with its values kept as written."""
    separator = " " if Path(path_items).suffix == ".item" else ","
    return pl.read_csv(path_items, separator=separator, infer_schema=False)
