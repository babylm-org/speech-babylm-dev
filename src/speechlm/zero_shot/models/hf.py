import math

import torch
from transformers import AutoFeatureExtractor, AutoModel

from speechlm.zero_shot.models.base import (
    SpeechLM,
    capture_inputs,
    name_transformer_layers,
)


class HFModel(SpeechLM):
    """A Hugging Face model with a CNN encoder and transformer layers (Wav2Vec2, HuBERT, WavLM...)."""

    def __init__(self, model_name: str, cache_dir=None):
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        preprocessor = AutoFeatureExtractor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        samples_per_frame = math.prod(model.config.conv_stride)
        frame_rate = preprocessor.sampling_rate // samples_per_frame
        super().__init__(model, preprocessor, frame_rate)

    @torch.no_grad()
    def get_representations(self, audio, attention_mask=None):
        with capture_inputs({"CNN": self.model.feature_projection}) as captured:
            outputs = self.model(
                audio, attention_mask=attention_mask, output_hidden_states=True
            )
        embeds, *layers = outputs.hidden_states
        return {
            "CNN": captured["CNN"],
            "embeds": embeds,
            **name_transformer_layers(layers),
        }
