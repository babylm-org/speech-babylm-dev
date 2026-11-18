"""ABX evaluation sets: extract a model's activations (get_activations.py), score every layer
on the set's ABX conditions (conditions.py, e.g. within- and across-speaker) with run_abx.py,
then delete the activations, so only one model and set is on disk at a time.

Data:
    triphone: LibriSpeech (openslr.org/12) with the ZeroSpeech 2021 item files
        audio  -> LIBRISPEECH_DIR/<split>/
        items  -> ZEROSPEECH_ITEM_DIR/item/triphone-<split>.item
        sample -> ZEROSPEECH_ITEM_DIR/sample/triphone-<split>-sample.item: 10 speakers
                  (5 F, 5 M) x 25 recordings, ~10% of the split, for the fast evaluation
    prosodic: English lexical stress, from the HaitongSUN/prosody-abx dataset: stress
              (recorded), stress_syn (Google TTS), stress_kokoro
        audio -> PROSODIC_AUDIO_DIR/<set>/<id>.wav, one word per file, 16 kHz mono
        items -> PROSODIC_ITEM_DIR/<set>.csv

Results: results/<model>/<task>/[fast/]abx_<condition id>.csv.
"""

import gc
import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import polars as pl
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

from speechlm.zero_shot.config import ABX_ITEM_DIR, ACTIVATION_DIR, DATA_DIR
from speechlm.zero_shot.tasks.abx.conditions import (
    PROSODIC_CONDITIONS,
    SEED,
    TRIPHONE_CONDITIONS,
    ABXCondition,
)
from speechlm.zero_shot.tasks.abx.get_activations import extract
from speechlm.zero_shot.tasks.abx.run_abx import evaluate_abx
from speechlm.zero_shot.tasks.base import EvalTask

SAMPLING_RATE = 16000

TRIPHONE_SPLITS = ("dev-clean", "dev-other")
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
LIBRISPEECH_DIR = DATA_DIR / "LibriSpeech"
ZEROSPEECH_ITEMS_URL = (
    "https://cognitive-ml.fr/downloads/phoneme-discovery/zerospeech2021-triphone.tar.gz"
)
ZEROSPEECH_ITEM_DIR = ABX_ITEM_DIR / "zerospeech2021-triphone"
SAMPLE_SPEAKERS_PER_SEX = 5
SAMPLE_FILES_PER_SPEAKER = 25

PROSODIC_HF_DATASET = "HaitongSUN/prosody-abx"
PROSODIC_SETS = {
    "stress": "english_stress/english_stress.parquet",
    "stress_syn": "english_stress_syn/english_stress_syn.parquet",
    "stress_kokoro": "english_stress_kokoro/english_stress_kokoro.parquet",
}
PROSODIC_AUDIO_DIR = DATA_DIR / "prosodic"
PROSODIC_ITEM_DIR = ABX_ITEM_DIR / "prosodic"


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
        layers=None,
        keep_activations=False,
        **_,
    ):
        """
        Args:
            batch_size: files per forward pass in extraction
            fast: score sample_items
            layers: score only these layers (e.g. ["CNN", "T6"]); all if None
            keep_activations: do not delete the activations afterwards
        """
        model_name = model.name
        path_items = self.sample_items if fast else self.path_items
        out_dir = self.results_dir(model, fast)
        conditions = [c for c in self.conditions if not (out_dir / f"abx_{c.id}.csv").exists()]
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
            if unknown := set(layers or []) - set(info["layers"]):
                msg = f"{model_name} has no layers {sorted(unknown)}; it has {info['layers']}"
                raise ValueError(msg)
            evaluate_abx(
                model_name,
                path_items,
                conditions,
                out_dir,
                activations_dir=ACTIVATION_DIR,
                layers=layers,
            )
        finally:
            # also on failure, so no activations are left behind
            if not keep_activations:
                shutil.rmtree(ACTIVATION_DIR / model_name, ignore_errors=True)


class TriphoneEval(ABXEval):
    """ZeroSpeech 2021 triphone ABX on a LibriSpeech split. Its sample is ~10% of the split."""

    file_format = "flac"
    conditions = TRIPHONE_CONDITIONS

    def __init__(self, split):
        self.split = split
        self.name = f"triphone-{split}"
        self.audio_dir = LIBRISPEECH_DIR / split
        self.path_items = ZEROSPEECH_ITEM_DIR / "item" / f"triphone-{split}.item"
        self.sample_items = ZEROSPEECH_ITEM_DIR / "sample" / f"triphone-{split}-sample.item"

    def setup(self):
        print("==> ZeroSpeech 2021 triphone items")
        if not (ZEROSPEECH_ITEM_DIR / "item").is_dir():
            archive = DATA_DIR / "zerospeech2021-triphone.tar.gz"
            download_and_extract(ZEROSPEECH_ITEMS_URL, archive, ZEROSPEECH_ITEM_DIR)
        print(f"==> LibriSpeech {self.split}")
        if not self.audio_dir.is_dir():
            url = LIBRISPEECH_URL.format(split=self.split)
            download_and_extract(url, DATA_DIR / f"{self.split}.tar.gz", DATA_DIR)
        self.make_fast()

    def make_fast(self):
        if not self.sample_items.exists():
            make_triphone_sample(self.split, self.path_items, self.sample_items)


class ProsodicEval(ABXEval):
    """Prosodic ABX (English stress) on one of its sets. The sets are small, so the sample is the
    whole set."""

    file_format = "wav"
    conditions = PROSODIC_CONDITIONS

    def __init__(self, set_name):
        self.name = set_name
        self.audio_dir = PROSODIC_AUDIO_DIR / set_name
        self.path_items = self.sample_items = PROSODIC_ITEM_DIR / f"{set_name}.csv"

    def setup(self):
        print(f"==> Prosodic ABX: {self.name}")
        if not self.path_items.exists():
            download_prosodic(self.name, self.audio_dir, self.path_items)


def read_items(path_items):
    """An item file with its values kept as written."""
    separator = " " if Path(path_items).suffix == ".item" else ","
    return pl.read_csv(path_items, separator=separator, infer_schema=False)


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


def make_triphone_sample(split, path_items, sample_path):
    """The item file restricted to a few speakers and recordings of the split."""
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
    items = read_items(path_items).filter(pl.col("speaker").is_in(selected))
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


def download_prosodic(set_name, audio_dir, item_path):
    """Write the set's words as 16 kHz mono wavs, and its item file."""
    parquet = hf_hub_download(PROSODIC_HF_DATASET, PROSODIC_SETS[set_name], repo_type="dataset")
    rows = pl.read_parquet(parquet).sort("id").to_dicts()

    audio_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for row in rows:
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if "target_onset" in row:  # recorded speech: cut the word out of its sentence
            audio = audio[round(row["target_onset"] * sr) : round(row["target_offset"] * sr)]
        if sr != SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        sf.write(audio_dir / f"{row['id']}.wav", audio.astype(np.float32), SAMPLING_RATE)
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
