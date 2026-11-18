import json
import math
from pathlib import Path

import soundfile as sf
import torch
import tqdm
from dotenv import load_dotenv

from speechlm.zero_shot.config import ACTIVATION_DIR
from speechlm.zero_shot.models.base import SpeechLM

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


def list_audio_files(
    audio_dir: str | Path, file_format: str = "flac", file_ids: set | None = None
) -> list[Path]:
    """The audio files in audio_dir (searched recursively), longest first, to keep padding small."""
    files = [
        f
        for f in sorted(Path(audio_dir).glob(f"**/*.{file_format}"))
        if file_ids is None or f.stem in file_ids
    ]
    return sorted(files, key=lambda f: sf.info(f).duration, reverse=True)


def extract(
    model: SpeechLM,
    audio_dir: str | Path,
    file_format: str = "flac",
    out_dir: str | Path = ACTIVATION_DIR,
    batch_size: int = 16,
    file_ids: set | None = None,
):
    """
    Save every layer's activations for the audio files in audio_dir (searched recursively)
    to <out_dir>/<model name>/<layer>/<file id>.pt, plus info.json for run_abx.py.

    Args:
        model (SpeechLM): A loaded model, from model_registry.get_model.
        file_ids (set | None): Only these files (names without extension); all if None.
        batch_size (int): Files per forward pass. Files are sorted by duration, so padding is
            small, but it still changes the activations slightly; 1 is exact.
    """
    model_dir = Path(out_dir) / model.name
    preprocessor = model.get_preprocessor()
    files = list_audio_files(audio_dir, file_format=file_format, file_ids=file_ids)
    layers = None

    for start in tqdm.trange(
        0, len(files), batch_size, desc=f"Extracting activations of {len(files)} files"
    ):
        batch_files = files[start : start + batch_size]
        signals, sampling_rates = zip(*(sf.read(f) for f in batch_files))
        assert len(set(sampling_rates)) == 1, f"Mixed sampling rates: {batch_files}"
        sampling_rate = sampling_rates[0]
        inputs = preprocessor(
            list(signals),
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
            padding_side="right",
        ).to(model.device)

        representations = model.get_representations(
            inputs.input_values, inputs.get("attention_mask")
        )
        layers = list(representations)
        for layer, activations in representations.items():
            layer_dir = model_dir / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            for audio_file, signal, act in zip(batch_files, signals, activations):
                # drop the frames that only cover batch padding
                duration = len(signal) / sampling_rate
                n_frames = min(act.shape[0], math.ceil(duration * model.frame_rate))
                # a copy, so the file holds only these frames, not the whole batch's storage
                torch.save(
                    act[:n_frames].cpu().clone(memory_format=torch.contiguous_format),
                    layer_dir / f"{audio_file.stem}.pt",
                )

    info = {"model": model.name, "frequency": model.frame_rate, "layers": layers}
    (model_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved activations to {model_dir}")
