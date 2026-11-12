"""ABX evaluation sets: extract a model's activations (get_activations.py), score every layer
on the set's ABX conditions (conditions.py, e.g. within- and across-speaker) with run_abx.py,
then delete the activations, so only one model and set is on disk at a time.

Results: results/<model>/abx/<item file name>/abx_<condition id>.csv.
"""

import gc
import json
import shutil
from pathlib import Path

import polars as pl
import torch

from speechlm.zero_shot.config import ACTIVATION_DIR, results_dir
from speechlm.zero_shot.tasks.abx.conditions import (
    LIBRISPEECH_DIR,
    PROSODIC_AUDIO_DIR,
    PROSODIC_CONDITIONS,
    PROSODIC_ITEM_DIR,
    TRIPHONE_CONDITIONS,
    ZEROSPEECH_ITEM_DIR,
    ABXCondition,
    ProsodicABX,
    TriphoneABX,
)
from speechlm.zero_shot.tasks.abx.get_activations import extract
from speechlm.zero_shot.tasks.abx.run_abx import evaluate_abx
from speechlm.zero_shot.tasks.base import EvalTask


class ABXEval(EvalTask):
    """An item file over a directory of audio, scored on some ABX conditions.

    sample_items: the item file of the fast evaluation
    """

    audio_dir: Path
    file_format: str
    path_items: Path
    sample_items: Path
    conditions: list[ABXCondition]

    def evaluate(
        self,
        model,
        batch_size=16,
        fast=False,
        every_n_layers=1,
        keep_activations=False,
        **_,
    ):
        """
        Args:
            batch_size: files per forward pass in extraction
            fast: score sample_items
            every_n_layers: score every n-th layer, always including the first and last
            keep_activations: do not delete the activations afterwards
        """
        model_name = model.name
        path_items = self.sample_items if fast else self.path_items
        out_dir = results_dir(model_name, "abx") / path_items.stem
        conditions = [
            c
            for c in self.conditions
            if not (out_dir / f"abx_{c.id}.csv").exists()
        ]
        if not conditions:
            print(f"Skipping {model_name} on {self.name}: results already in {out_dir}")
            return

        try:
            extract(
                model,
                self.audio_dir,
                file_format=self.file_format,
                out_dir=ACTIVATION_DIR,
                batch_size=batch_size,
                file_ids=set(read_items(path_items)["#file"]),
            )
            gc.collect()
            torch.cuda.empty_cache()
            info = json.loads((ACTIVATION_DIR / model_name / "info.json").read_text())
            evaluate_abx(
                model_name,
                path_items,
                conditions,
                activations_dir=ACTIVATION_DIR,
                layers=select_layers(info["layers"], every_n_layers),
            )
        finally:
            # also on failure, so no activations are left behind
            if not keep_activations:
                shutil.rmtree(ACTIVATION_DIR / model_name, ignore_errors=True)


class TriphoneEval(ABXEval):
    """ZeroSpeech 2021 triphone ABX on a LibriSpeech split. Its sample is ~10% of the split
    (see TriphoneABX)."""

    file_format = "flac"
    conditions = TRIPHONE_CONDITIONS

    def __init__(self, split):
        self.split = split
        self.name = f"triphone-{split}"
        self.audio_dir = LIBRISPEECH_DIR / split
        self.path_items = ZEROSPEECH_ITEM_DIR / "item" / f"triphone-{split}.item"
        self.sample_items = (
            ZEROSPEECH_ITEM_DIR / "sample" / f"triphone-{split}-sample.item"
        )

    def download(self):
        TriphoneABX.download([self.split])


class ProsodicEval(ABXEval):
    """Prosodic ABX (English stress) on one of its sets. The sets are small, so the sample is the
    whole set."""

    file_format = "wav"
    conditions = PROSODIC_CONDITIONS

    def __init__(self, set_name):
        self.name = set_name
        self.audio_dir = PROSODIC_AUDIO_DIR / set_name
        self.path_items = self.sample_items = PROSODIC_ITEM_DIR / f"{set_name}.csv"

    def download(self):
        ProsodicABX.download([self.name])


def read_items(path_items):
    """An item file with its values kept as written."""
    separator = " " if Path(path_items).suffix == ".item" else ","
    return pl.read_csv(path_items, separator=separator, infer_schema=False)


def select_layers(layers, every_n_layers=1):
    """Every n-th layer, always including the first and the last."""
    selected = layers[::every_n_layers]
    if layers[-1] not in selected:
        selected.append(layers[-1])
    return selected
