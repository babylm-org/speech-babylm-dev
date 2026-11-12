from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
from transformers.feature_extraction_utils import BatchFeature

from speechlm.zero_shot.config import CHECKPOINT_REPO
from speechlm.zero_shot.models.base import (
    SpeechLM,
    capture_inputs,
    name_transformer_layers,
)
from speechlm.zero_shot.models.vendors.melhubert.model import (
    MelHuBERTConfig,
    MelHuBERTModel,
)

MELHUBERT_DIR = Path(__file__).parent / "vendors" / "melhubert"


class MelPreprocessor:
    """Normalized log Mel features of 20 ms frames, zero-padded, with an attention mask."""

    def __init__(self):
        mean, std = np.load(MELHUBERT_DIR / "libri-960-mean-std.npy")
        self.mean = torch.as_tensor(mean.reshape(-1), dtype=torch.float32)
        self.std = torch.as_tensor(std.reshape(-1), dtype=torch.float32)

    def __call__(self, signals, sampling_rate, **_):
        if sampling_rate != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sampling_rate} Hz")
        if isinstance(signals, np.ndarray):
            signals = [signals]
        features = [self._features(signal) for signal in signals]
        input_values = pad_sequence(features, batch_first=True)
        attention_mask = torch.zeros(input_values.shape[:2])
        for i, feature in enumerate(features):
            attention_mask[i, : len(feature)] = 1
        return BatchFeature(
            {"input_values": input_values, "attention_mask": attention_mask}
        )

    def _features(self, signal):
        """Adapted from extract_fbank in
        https://github.com/nervjack2/MelHuBERT/blob/main/extract_feature.py"""
        waveform = torch.as_tensor(signal * 2**15, dtype=torch.float32).unsqueeze(0)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=40,
            sample_frequency=16000,
            window_type="hamming",
            frame_length=25,
            frame_shift=10,
        )
        fbank = (fbank - self.mean) / self.std
        # every two 10 ms frames become one 20 ms frame
        odd, even = fbank[::2], fbank[1::2]
        if len(even) < len(odd):
            even = torch.cat([even, torch.zeros(1, even.shape[1])])
        return torch.cat([odd, even], dim=1)


class MelHuBERT(SpeechLM):
    """MelHuBERT (https://github.com/nervjack2/MelHuBERT), a HuBERT on Mel spectrograms.

    It has no CNN, so its first representation is "embeds".
    """

    def __init__(self, model_dir):
        config_file = MELHUBERT_DIR / "config_model_20ms.yaml"
        config = yaml.safe_load(config_file.read_text())["melhubert"]
        model = MelHuBERTModel(MelHuBERTConfig(config))
        checkpoint = hf_hub_download(
            CHECKPOINT_REPO,
            "melhubert/melhubert_960_stage2_20ms.safetensors",
            cache_dir=model_dir,
        )
        model.load_state_dict(load_file(checkpoint))
        super().__init__(model, MelPreprocessor(), frame_rate=50)

    @torch.no_grad()
    def get_representations(self, audio, attention_mask=None):
        with capture_inputs({"embeds": self.model.encoder.layers[0]}) as captured:
            *_, layers, _ = self.model(
                audio, attention_mask, no_pred=True, get_hidden=True
            )
        # the transformer takes (frames, batch, dim)
        return {
            "embeds": captured["embeds"].transpose(0, 1),
            **name_transformer_layers(layers),
        }
