"""Build the evaluation data from the original sources, and upload it to EVAL_REPO.

Run once by the maintainers; everyone else downloads EVAL_REPO with prepare_eval, so they all
evaluate on the same data. For each task (see EvalTask), it writes in EVAL_DATA_DIR:
    full_eval/<family>/<subset>.zip and its data file          the whole set
    fast_eval/<family>_fast/<subset>.zip and its data file     the fixed subset of --fast

    triphone: LibriSpeech (openslr.org/12) with the ZeroSpeech 2021 item files; fast: 10
              speakers (5 F, 5 M) x 25 recordings, ~10% of the split
    prosodic: English lexical stress (HaitongSUN/prosody-abx); fast: the whole set
    babyslm:  BabySLM's minimal pairs, dev split; fast: 10% of the pairs (of each type, for
              syntactic), every voice kept

The original sources are downloaded to RAW_DATA_DIR.

Usage:
    python -m scripts.eval.build_eval_data                      # build every task
    python -m scripts.eval.build_eval_data --tasks stress       # build some
    python -m scripts.eval.build_eval_data --upload             # and upload (needs HF_TOKEN)
"""

import argparse
import io
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import polars as pl
import soundfile as sf
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from speechlm.zero_shot.config import EVAL_DATA_DIR, EVAL_REPO, RAW_DATA_DIR
from speechlm.zero_shot.task_registry import TASKS
from speechlm.zero_shot.tasks.abx.task import ProsodicEval, TriphoneEval, read_items
from speechlm.zero_shot.tasks.babyslm.task import BabySLMEval

load_dotenv()  # HF_TOKEN

SEED = 0
SAMPLING_RATE = 16000

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
ZEROSPEECH_ITEMS_URL = (
    "https://cognitive-ml.fr/downloads/phoneme-discovery/zerospeech2021-triphone.tar.gz"
)
TRIPHONE_SPEAKERS_PER_SEX = 5
TRIPHONE_FILES_PER_SPEAKER = 25

PROSODIC_HF_DATASET = "HaitongSUN/prosody-abx"
PROSODIC_SETS = {
    "stress": "english_stress/english_stress.parquet",
    "stress_syn": "english_stress_syn/english_stress_syn.parquet",
    "stress_kokoro": "english_stress_kokoro/english_stress_kokoro.parquet",
}

BABYSLM_URL = "https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/{task}/dev.zip"
BABYSLM_SHARED = {"lexical": ["word", "frequency", "length"], "syntactic": ["type", "subtype"]}
BABYSLM_DIFFERS = {"lexical": ["phones"], "syntactic": ["transcription"]}
BABYSLM_FAST_FRACTION = 0.1


def build_triphone(task: TriphoneEval):
    split = task.subset
    item_dir = RAW_DATA_DIR / "zerospeech2021-triphone"
    if not (item_dir / "item").is_dir():
        archive = RAW_DATA_DIR / "zerospeech2021-triphone.tar.gz"
        download_and_extract(ZEROSPEECH_ITEMS_URL, archive, item_dir)
    audio_dir = RAW_DATA_DIR / "LibriSpeech" / split
    if not audio_dir.is_dir():
        url = LIBRISPEECH_URL.format(split=split)
        download_and_extract(url, RAW_DATA_DIR / f"{split}.tar.gz", RAW_DATA_DIR)

    items = read_items(item_dir / "item" / f"triphone-{split}.item")
    audio = {f.stem: f for f in audio_dir.rglob("*.flac")}
    written = []
    for fast, subset in [(False, items), (True, triphone_sample(items, split))]:
        task.data_dir(fast).mkdir(parents=True, exist_ok=True)
        subset.write_csv(task.items_file(fast), separator=" ")
        files = [audio[f] for f in subset["#file"].unique().sort()]
        zip_audio(files, task.data_dir(fast) / f"{split}.zip")
        written += [task.items_file(fast), task.data_dir(fast) / f"{split}.zip"]
        print(f"    {'fast' if fast else 'full'}: {len(subset)} triphones, {len(files)} recordings")
    return written


def triphone_sample(items, split):
    """The items of a few speakers and recordings of the split."""
    # SPEAKERS.TXT lines: "ID | SEX | SUBSET | MINUTES | NAME"
    speakers = (
        pl.read_csv(
            RAW_DATA_DIR / "LibriSpeech" / "SPEAKERS.TXT",
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
        .sample(TRIPHONE_SPEAKERS_PER_SEX, seed=SEED)
        .sort()
        .to_list()
    ]
    items = items.filter(pl.col("speaker").is_in(selected))
    files = [
        f
        for speaker in selected
        for f in items.filter(pl.col("speaker") == speaker)["#file"]
        .unique()
        .sort()
        .sample(TRIPHONE_FILES_PER_SPEAKER, seed=SEED)
        .to_list()
    ]
    return items.filter(pl.col("#file").is_in(files))


def build_prosodic(task: ProsodicEval):
    audio_dir = RAW_DATA_DIR / "prosodic" / task.subset
    items = convert_prosodic(task.subset, audio_dir)
    files = sorted(audio_dir.glob("*.wav"))
    written = []
    for fast in (False, True):  # the sets are small, so the fast subset is the whole set
        task.data_dir(fast).mkdir(parents=True, exist_ok=True)
        items.write_csv(task.items_file(fast))
        zip_audio(files, task.data_dir(fast) / f"{task.subset}.zip")
        written += [task.items_file(fast), task.data_dir(fast) / f"{task.subset}.zip"]
    n_speakers = items["speaker"].n_unique()
    print(f"    {len(items)} words from {n_speakers} speakers")
    return written


def convert_prosodic(set_name, audio_dir):
    """Write the set's words to audio_dir as 16 kHz mono wavs; return its items."""
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
    return pl.DataFrame(items)


def build_babyslm(task: BabySLMEval):
    raw_dir = RAW_DATA_DIR / "babyslm" / task.subset
    if not (raw_dir / "gold.csv").exists():
        download_babyslm(task.subset, raw_dir)
    pairs = babyslm_pairs(task.subset, pd.read_csv(raw_dir / "gold.csv"))
    written = []
    for fast, subset in [(False, pairs), (True, babyslm_sample(pairs, task.subset))]:
        task.data_dir(fast).mkdir(parents=True, exist_ok=True)
        data_file = task.data_dir(fast) / f"{task.subset}.csv"
        subset.to_csv(data_file, index=False)
        names = sorted(pd.unique(subset[["filename_correct", "filename_wrong"]].to_numpy().ravel()))
        zip_audio(
            [raw_dir / "wavs" / f"{n}.wav" for n in names],
            task.data_dir(fast) / f"{task.subset}.zip",
        )
        written += [data_file, task.data_dir(fast) / f"{task.subset}.zip"]
        print(f"    {'fast' if fast else 'full'}: {len(subset)} pairs, {len(names)} files")
    return written


def download_babyslm(task, out_dir):
    """The dev split's gold.csv and wavs/<filename>.wav, as distributed."""
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    url = BABYSLM_URL.format(task=task)
    print(f"    downloading {url}")
    # extracted next to out_dir, so the files are moved rather than copied
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        archive = f"{tmp}/dev.zip"
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        # the archive's folder is named dev or dev_16
        (gold,) = Path(tmp).rglob("gold.csv")
        for wav in gold.parent.glob("*.wav"):
            wav.rename(out_dir / "wavs" / wav.name)
        gold.rename(out_dir / "gold.csv")


def babyslm_pairs(task, gold):
    """One row per minimal pair and voice: its correct and wrong files side by side."""
    differs = ["filename", *BABYSLM_DIFFERS[task]]
    correct = gold[gold["correct"] == 1][["id", "voice", *BABYSLM_SHARED[task], *differs]]
    wrong = gold[gold["correct"] == 0][["id", "voice", *differs]]
    pairs = correct.merge(
        wrong, on=["id", "voice"], suffixes=("_correct", "_wrong"), validate="1:1"
    )
    if 2 * len(pairs) != len(gold):
        msg = f"{task}: not every file is in exactly one pair"
        raise ValueError(msg)
    columns = [f"{c}_{side}" for c in differs for side in ("correct", "wrong")]
    return pairs[["id", "voice", *BABYSLM_SHARED[task], *columns]].sort_values(["id", "voice"])


def babyslm_sample(pairs, task):
    """A fraction of the minimal pairs, with all their voices."""
    ids = pairs.drop_duplicates("id")
    if task == "syntactic":  # so every type keeps its share
        ids = ids.groupby("type")
    ids = ids.sample(frac=BABYSLM_FAST_FRACTION, random_state=SEED)["id"]
    return pairs[pairs["id"].isin(ids)]


def download_and_extract(url, archive, dest):
    """Download a .tar.gz (unless already there) and extract it into dest."""
    if not archive.exists():
        print(f"    downloading {url}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        # via a .part file, so an interrupted download is not mistaken for a finished one
        part = archive.with_name(archive.name + ".part")
        with urllib.request.urlopen(url) as response, part.open("wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            done, last_shown = 0, -1
            while chunk := response.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and 10 * done // total != last_shown:
                    last_shown = 10 * done // total
                    print(f"    {100 * done // total}% of {total / 1e6:.0f} MB", flush=True)
        part.rename(archive)
    print(f"    extracting {archive.name}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")


def zip_audio(files, zip_path):
    """The audio files, flat (one folder level), in zip_path."""
    part = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)
    part.rename(zip_path)


BUILDERS = {TriphoneEval: build_triphone, ProsodicEval: build_prosodic, BabySLMEval: build_babyslm}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--upload", action="store_true", help=f"Upload them to {EVAL_REPO}")
    args = parser.parse_args()

    written = []
    for name in args.tasks:
        task = TASKS[name]
        print(f"==> {name}")
        written += BUILDERS[type(task)](task)

    if args.upload:
        api = HfApi()
        api.create_repo(EVAL_REPO, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=EVAL_REPO,
            repo_type="dataset",
            folder_path=EVAL_DATA_DIR,
            path_in_repo="evaluation_data",
            allow_patterns=[f.relative_to(EVAL_DATA_DIR).as_posix() for f in written],
            commit_message=f"evaluation data of {', '.join(args.tasks)}",
        )
        print(f"Uploaded {len(written)} files to {EVAL_REPO}")


if __name__ == "__main__":
    main()
