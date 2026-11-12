"""ABX conditions: ZeroSpeech 2021 triphone and Prosodic ABX (English stress).

Each benchmark is a subclass of ABXCondition, and its instances are its conditions: within or
across speakers, and the random baseline, e.g. TriphoneABX("within") or
ProsodicABX("across", random=True).
"""

import io
import math
import tarfile
import urllib.request
from decimal import Decimal
from pathlib import Path

import librosa
import numpy as np
import polars as pl
import soundfile as sf
import torch
from fastabx import Dataset, InMemoryAccessor, Score, Subsampler, Task
from fastabx.dataset import read_labels
from huggingface_hub import hf_hub_download

from speechlm.zero_shot.config import DATA_DIR, ITEM_DIR

SEED = 0
SAMPLING_RATE = 16000

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
LIBRISPEECH_DIR = DATA_DIR / "LibriSpeech"
ZEROSPEECH_ITEMS_URL = (
    "https://cognitive-ml.fr/downloads/phoneme-discovery/zerospeech2021-triphone.tar.gz"
)
ZEROSPEECH_ITEM_DIR = ITEM_DIR / "zerospeech2021-triphone"
SAMPLE_SPEAKERS_PER_SEX = 5
SAMPLE_FILES_PER_SPEAKER = 25
PROSODIC_HF_DATASET = "HaitongSUN/prosody-abx"
PROSODIC_AUDIO_DIR = DATA_DIR / "prosodic"
PROSODIC_ITEM_DIR = ITEM_DIR / "prosodic"


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

    @classmethod
    def download(cls):
        """Fetch the benchmark's audio and item files. Safe to re-run: what exists is skipped."""
        raise NotImplementedError

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
    """ZeroSpeech 2021 triphone ABX, on LibriSpeech (openslr.org/12) with fastabx item files.

    audio  -> LIBRISPEECH_DIR/<split>/
    items  -> ZEROSPEECH_ITEM_DIR/item/triphone-<split>.item
    sample -> ZEROSPEECH_ITEM_DIR/sample/triphone-<split>-sample.item: 10 speakers (5 F, 5 M) x
              25 recordings, ~10% of the split, so every layer can be scored quickly
    """

    name = "zero_shot_triphone"
    on = "#phone"
    by = ["next-phone", "prev-phone"]
    # average over contexts first, then over speakers, as in ZeroSpeech
    levels = [("next-phone", "prev-phone"), "speaker"]
    splits = ("dev-clean", "dev-other")

    @classmethod
    def download(cls, splits=None):
        print("==> ZeroSpeech 2021 triphone items")
        if not (ZEROSPEECH_ITEM_DIR / "item").is_dir():
            archive = DATA_DIR / "zerospeech2021-triphone.tar.gz"
            download_and_extract(ZEROSPEECH_ITEMS_URL, archive, ZEROSPEECH_ITEM_DIR)
        for split in splits or cls.splits:
            print(f"==> LibriSpeech {split}")
            if not (LIBRISPEECH_DIR / split).is_dir():
                url = LIBRISPEECH_URL.format(split=split)
                download_and_extract(url, DATA_DIR / f"{split}.tar.gz", DATA_DIR)
            make_zerospeech_sample(split)

    def subsampler(self):
        # ZeroSpeech 2021: at most 10 A, B or X per cell, and 5 X per (A, B) across speakers
        max_x_across = 5 if self.speakers == "across" else None
        return Subsampler(max_size_group=10, max_x_across=max_x_across, seed=SEED)


class ProsodicABX(ABXCondition):
    """Prosodic ABX (https://arxiv.org/abs/2604.02102), as in the prosodic-abx repo's run_abx.py:
    stress pattern ON, word BY, no subsampling.

    English lexical stress, from the HaitongSUN/prosody-abx dataset: stress (recorded),
    stress_syn (Google TTS), stress_kokoro.

    audio -> PROSODIC_AUDIO_DIR/<set>/<id>.wav, one word per file, 16 kHz mono
    items -> PROSODIC_ITEM_DIR/<set>.csv
    """

    name = "prosodic"
    on = "accent_pattern"
    by = ["phone_sequence"]
    levels = ["speaker"]
    sets = {
        "stress": "english_stress/english_stress.parquet",
        "stress_syn": "english_stress_syn/english_stress_syn.parquet",
        "stress_kokoro": "english_stress_kokoro/english_stress_kokoro.parquet",
    }

    @classmethod
    def download(cls, sets=None):
        for name in sets or cls.sets:
            print(f"==> Prosodic ABX: {name}")
            audio_dir = PROSODIC_AUDIO_DIR / name
            item_path = PROSODIC_ITEM_DIR / f"{name}.csv"
            if item_path.exists():
                continue
            parquet = hf_hub_download(
                PROSODIC_HF_DATASET, cls.sets[name], repo_type="dataset"
            )
            rows = pl.read_parquet(parquet).sort("id").to_dicts()

            audio_dir.mkdir(parents=True, exist_ok=True)
            items = []
            for row in rows:
                audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if "target_onset" in row:  # recorded speech: cut the word out of its sentence
                    audio = audio[
                        round(row["target_onset"] * sr) : round(row["target_offset"] * sr)
                    ]
                if sr != SAMPLING_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
                sf.write(
                    audio_dir / f"{row['id']}.wav",
                    audio.astype(np.float32),
                    SAMPLING_RATE,
                )
                # the prosodic-abx repo's item columns: the word is phone_sequence,
                # the stress pattern accent_pattern
                items.append(
                    {
                        "#file": row["id"],
                        "onset": 0.0,
                        "offset": len(audio) / SAMPLING_RATE,
                        "phone_sequence": row["target"],
                        "accent_pattern": row["label"],
                        "speaker": row["speaker"],
                        "lexical_category": row["lexical_category"],
                    }
                )
            item_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(items).write_csv(item_path)
            n_speakers = len({item["speaker"] for item in items})
            print(f"    {len(items)} words from {n_speakers} speakers -> {item_path}")

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
            end = min(
                math.floor(row["offset"] * freq - half) + 1, features[file_id].shape[0]
            )
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


def download_and_extract(url, archive, dest):
    """Download a .tar.gz (unless already there) and extract it into dest."""
    if not archive.exists():
        print(f"    downloading {url}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        # via a .part file, so an interrupted download is not mistaken for a finished one
        part = archive.with_name(archive.name + ".part")
        with urllib.request.urlopen(url) as response, open(part, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            done, last_shown = 0, -1
            while chunk := response.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and 10 * done // total != last_shown:
                    last_shown = 10 * done // total
                    print(
                        f"    {100 * done // total}% of {total / 1e6:.0f} MB",
                        flush=True,
                    )
        part.rename(archive)
    print(f"    extracting {archive.name}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")


def make_zerospeech_sample(split):
    """The ZeroSpeech item file restricted to a few speakers and recordings."""
    sample_path = ZEROSPEECH_ITEM_DIR / "sample" / f"triphone-{split}-sample.item"
    if sample_path.exists():
        return
    # SPEAKERS.TXT lines: "ID | SEX | SUBSET | MINUTES | NAME"
    speakers = (
        pl.read_csv(
            LIBRISPEECH_DIR / "SPEAKERS.TXT",
            separator="|",
            comment_prefix=";",
            has_header=False,
            new_columns=["id", "sex", "subset", "minutes", "name"],
            infer_schema=False,
            truncate_ragged_lines=True,
        )
        .with_columns(pl.all().str.strip_chars())
        .filter(pl.col("subset") == split)
    )
    selected = [
        speaker
        for sex in ["F", "M"]
        for speaker in speakers.filter(pl.col("sex") == sex)["id"]
        .sample(SAMPLE_SPEAKERS_PER_SEX, seed=SEED)
        .sort()
        .to_list()
    ]
    items = pl.read_csv(
        ZEROSPEECH_ITEM_DIR / "item" / f"triphone-{split}.item",
        separator=" ",
        infer_schema=False,
    ).filter(pl.col("speaker").is_in(selected))
    files = [
        f
        for speaker in selected
        for f in items.filter(pl.col("speaker") == speaker)["#file"]
        .unique()
        .sort()
        .sample(SAMPLE_FILES_PER_SPEAKER, seed=SEED)
        .to_list()
    ]
    items = items.filter(pl.col("#file").is_in(files))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    items.write_csv(sample_path, separator=" ")
    print(
        f"    sample: {len(items)} triphones, {len(files)} recordings, speakers {selected} -> {sample_path}"
    )


TRIPHONE_CONDITIONS = [
    TriphoneABX("within"),
    TriphoneABX("across"),
    TriphoneABX("within", random=True),
]
PROSODIC_CONDITIONS = [
    ProsodicABX("across"),
    ProsodicABX("across", random=True),
]
CONDITIONS = {c.id: c for c in TRIPHONE_CONDITIONS + PROSODIC_CONDITIONS}
