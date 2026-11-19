import torch
from huggingface_hub import hf_hub_download
from torchaudio.models import hubert_pretrain_base
from transformers import Wav2Vec2FeatureExtractor

from speechlm.zero_shot.models.base import (
    SpeechLM,
    capture_inputs,
    name_transformer_layers,
)


class TorchaudioHuBERT(SpeechLM):
    """A model whose self.model is a torchaudio Wav2Vec2Model (HuBERT, wav2vec 2.0)."""

    @torch.no_grad()
    def get_representations(self, audio, attention_mask=None):
        encoder = self.model.encoder
        modules = {
            "CNN": encoder.feature_projection,
            "embeds": encoder.transformer.layers[0],
        }
        with capture_inputs(modules) as captured:
            layers, _ = self.model.extract_features(audio)
        return {**captured, **name_transformer_layers(layers)}


class BabyHuBERT(TorchaudioHuBERT):
    """BabyHuBERT (https://huggingface.co/MarvinLvn/BabyHuBERT), a torchaudio HuBERT."""

    def __init__(self, model_dir):
        # as in the model card
        model = hubert_pretrain_base(num_classes=500)
        checkpoint = hf_hub_download("MarvinLvn/BabyHuBERT", "BabyHuBERT.ckpt", cache_dir=model_dir)
        state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
        model.load_state_dict({name.replace("model.", ""): v for name, v in state_dict.items()})
        super().__init__("BabyHuBERT", model.wav2vec2, Wav2Vec2FeatureExtractor(), frame_rate=50)
