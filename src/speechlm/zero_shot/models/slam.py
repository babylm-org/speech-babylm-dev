"""Slam (https://arxiv.org/abs/2502.15814), a speech language model over mHuBERT units, run
with slamkit (https://github.com/slp-rl/slamkit). Checkpoints: huggingface.co/collections/slprl/slam
"""

import math
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from slamkit.feature_extractor import HubertFeatureExtractor
from slamkit.model import UnitLM
from slamkit.model.speech_lm import SpeechLM as SlamkitSpeechLM
from slamkit.tokeniser import UnitTokeniser
from transformers import Wav2Vec2FeatureExtractor

from speechlm.zero_shot.config import CHECKPOINT_REPO
from speechlm.zero_shot.models.base import SpeechLM
from speechlm.zero_shot.models.hf import HFModel

# slamkit's unit_hubert_25 tokeniser (config/tokeniser/unit_hubert_25.yaml), used by every
# Slam checkpoint: layer 11 of mHuBERT-base-25Hz, quantized into 500 units, repeats merged.
# The k-means comes from our mirror: its dl.fbaipublicfiles.com original had an expired
# TLS certificate in November 2026
ENCODER = "slprl/mhubert-base-25hz"
KMEANS = "slam/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin"
LAYER = 11
NUM_UNITS = 500


class Slam(HFModel):
    """A Slam checkpoint: slprl/slam, slprl/slam_scaled or slprl/slam_large.

    get_log_probs is slamkit's mean log likelihood per unit, as in its sWUGGY and sBLIMP.
    Representations are those of the speech encoder, mHuBERT-base-25Hz, at 25 Hz.
    """

    def __init__(self, model_name, model_dir):
        encoder_dir, lm_dir = download(model_name, model_dir)
        feature_extractor = HubertFeatureExtractor(
            pretrained_model=str(encoder_dir),
            # only used when kmeans_model.bin is missing, which download() prevents
            kmeans_path=f"https://huggingface.co/{CHECKPOINT_REPO}/resolve/main/{KMEANS}",
            layer=LAYER,
            num_units=NUM_UNITS,
            cache_path=str(kmeans_cache(model_dir)),
        )
        tokeniser = UnitTokeniser(
            feature_extractor,
            dedup=True,
            bos_eos_token_id=1,
            pad_token_id=0,
            num_units=NUM_UNITS,
        )
        # slamkit computes in bfloat16, whose log probabilities often tie for short words;
        # the weights are bfloat16 values, so float32 keeps them exactly
        lm = UnitLM.from_pretrained(str(lm_dir)).float()
        self.slamkit_lm = SlamkitSpeechLM(lm, tokeniser, device="cpu")
        encoder = feature_extractor.model
        # raw waveforms, as slamkit reads them (the encoder's preprocessor_config would
        # normalize them); the attention mask gives get_log_probs the lengths
        preprocessor = Wav2Vec2FeatureExtractor(
            do_normalize=False, return_attention_mask=True
        )
        samples_per_frame = math.prod(encoder.config.conv_stride)
        frame_rate = preprocessor.sampling_rate // samples_per_frame
        # skips HFModel.__init__, which loads its own model and preprocessor
        SpeechLM.__init__(self, encoder, preprocessor, frame_rate)

    def to(self, device):
        # the LM and the tokeniser, which holds the encoder
        self.slamkit_lm.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def get_log_probs(self, audio, attention_mask=None):
        """slamkit's SpeechLM.log_likelihood, except that each utterance is tokenised on its
        own: the encoder's group norm would also normalize the batch padding."""
        if attention_mask is None:
            lengths = [audio.shape[1]] * len(audio)
        else:
            lengths = attention_mask.sum(dim=1).tolist()
        tokeniser = self.slamkit_lm.tokeniser
        units = [
            tokeniser.audio_represent(signal[None, :length])[0]
            for signal, length in zip(audio, lengths, strict=True)
        ]
        tokeniser.text_tokeniser.padding_side = "right"
        tokens = tokeniser.string_tokenise(
            tokeniser.stringify_representation(units),
            return_tensors="pt",
            padding=True,
        )["input_ids"].to(self.slamkit_lm.device)
        return self.slamkit_lm.model.log_likelihood(tokens, mean_nll=True)


def download(model_name, model_dir):
    """Fetch the k-means, the encoder and the LM; return the encoder's and the LM's
    directories."""
    # slamkit looks for the k-means at <cache_path>/kmeans_model.bin
    kmeans_file = kmeans_cache(model_dir) / "kmeans_model.bin"
    if not kmeans_file.exists():
        kmeans_file.parent.mkdir(parents=True, exist_ok=True)
        mirror = hf_hub_download(CHECKPOINT_REPO, KMEANS, cache_dir=model_dir)
        shutil.copy(mirror, kmeans_file)
    return (
        Path(snapshot_download(ENCODER, cache_dir=model_dir)),
        Path(snapshot_download(model_name, cache_dir=model_dir)),
    )


def kmeans_cache(model_dir):
    return Path(model_dir) / "slamkit"
