"""ABX conditions: ZeroSpeech 2021 triphone and Prosodic ABX (English stress).

Each benchmark is a subclass of ABXCondition, and its instances are its conditions: within or
across speakers, and the random baseline, e.g. TriphoneABX("within") or
ProsodicABX("across", random=True). Their data is built by scripts/eval/build_eval_data.py.
"""

import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import torch
from fastabx import Dataset, InMemoryAccessor, Score, Subsampler, Task
from fastabx.dataset import read_labels

SEED = 0


class ABXCondition:
    """A condition of an ABX benchmark; the benchmark is defined by the class attributes:

    name: the start of the task's id, used to name its results
    on: the label that differs between A and B; X shares it with A
    by: the labels that A, B and X have in common
    levels: how the contrasts' scores are averaged
    """

    name: str
    on: str
    by: list[str]
    levels: list

    def __init__(self, speakers="across", random=False):
        """
        Args:
            speakers: "within" for A, B and X from the same speaker, "across" for A and B from
                one speaker and X from another.
            random: A control with the ON labels shuffled within each cell: the same cells as the
                real task, but whether A and X share a category is random, so the expected
                accuracy is 0.5.
        """
        if speakers not in ("within", "across"):
            raise ValueError(f'speakers must be "within" or "across", not {speakers!r}')
        self.speakers = speakers
        self.random = random

    @property
    def id(self):
        return f"{self.name}_{self.speakers}" + ("_random" if self.random else "")

    @property
    def label(self):
        """The label of this condition in the results and plots."""
        return "random" if self.random else f"{self.speakers}-speaker"

    def load_dataset(self, path_items, features_dir, frequency=50):
        """fastabx Dataset from an item file and features_dir/<file id>.pt of shape (n_frames, dim)."""
        return Dataset.from_item(path_items, features_dir, frequency=frequency)

    def subsampler(self):
        return None

    def score(self, dataset):
        """Return one row per contrast (e.g. phone pair A, B) with its labels, "score" (its ABX
        error rate) and "size"; the mean score is the overall error rate."""
        within = self.speakers == "within"
        by = ["speaker", *self.by] if within else self.by
        if self.random:
            dataset = shuffle_labels(dataset, self.on, by)
        task = Task(
            dataset,
            on=self.on,
            by=by,
            across=None if within else ["speaker"],
            subsampler=self.subsampler(),
        )
        return Score(task, "angular").details(levels=self.levels).to_dicts()


class TriphoneABX(ABXCondition):
    """ZeroSpeech 2021 triphone ABX, on LibriSpeech with fastabx item files."""

    name = "zero_shot_triphone"
    on = "#phone"
    by = ["next-phone", "prev-phone"]
    # average over contexts first, then over speakers, as in ZeroSpeech
    levels = [("next-phone", "prev-phone"), "speaker"]

    def subsampler(self):
        # ZeroSpeech 2021: at most 10 A, B or X per cell, and 5 X per (A, B) across speakers
        max_x_across = 5 if self.speakers == "across" else None
        return Subsampler(max_size_group=10, max_x_across=max_x_across, seed=SEED)


class ProsodicABX(ABXCondition):
    """Prosodic ABX (https://arxiv.org/abs/2604.02102), as in the prosodic-abx repo's run_abx.py:
    stress pattern ON, word BY, no subsampling.
    """

    name = "prosodic"
    on = "accent_pattern"
    by = ["phone_sequence"]
    levels = ["speaker"]

    def load_dataset(self, path_items, features_dir, frequency=50):
        """Segments running past the end of their features are cut at the last frame.

        Whole-word items often end one frame past the model's output, which Dataset.from_item
        rejects; the prosodic-abx repo clamps them the same way.
        """
        labels = read_labels(path_items, "#file", "onset", "offset")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features, segments, indices, pos = {}, [], {}, 0
        for i, row in enumerate(labels.iter_rows(named=True)):
            file_id = row["#file"]
            if file_id not in features:
                features[file_id] = torch.load(Path(features_dir) / f"{file_id}.pt")
            # fastabx's item_frontiers, with the end clamped
            freq, half = Decimal(str(frequency)), Decimal("0.5")
            start = math.ceil(row["onset"] * freq - half)
            end = min(math.floor(row["offset"] * freq - half) + 1, features[file_id].shape[0])
            segments.append(features[file_id][start:end])
            indices[i] = (pos, pos + end - start)
            pos += end - start
        return Dataset(
            labels=labels,
            accessor=InMemoryAccessor(indices, torch.cat(segments), device),
        )


def shuffle_labels(dataset, on, by):
    """The dataset with the ON labels shuffled within each BY group."""
    # an independent random key per row, so each group gets its own permutation;
    # pl.col(on).shuffle(seed).over(by) applies the same one to every group of the same size
    key = np.random.default_rng(SEED).random(len(dataset.labels))
    labels = (
        dataset.labels.with_columns(_shuffle_key=pl.Series(key))
        .with_columns(pl.col(on).sort_by("_shuffle_key").over(by))
        .drop("_shuffle_key")
    )
    return Dataset(labels=labels, accessor=dataset.accessor)


TRIPHONE_CONDITIONS = [
    TriphoneABX("within"),
    TriphoneABX("across"),
    TriphoneABX("within", random=True),
]
PROSODIC_CONDITIONS = [
    ProsodicABX("across"),
    ProsodicABX("across", random=True),
]
