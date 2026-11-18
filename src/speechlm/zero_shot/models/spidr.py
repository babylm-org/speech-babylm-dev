import torch
from transformers import Wav2Vec2FeatureExtractor

from speechlm.zero_shot.models.base import (
    SpeechLM,
    capture_inputs,
    name_transformer_layers,
)


class SpidR(SpeechLM):
    """SpidR (https://github.com/facebookresearch/spidr), loaded through torch.hub."""

    def __init__(self):
        # fetches the code and weights into torch.hub's own cache, not models/
        model = torch.hub.load("facebookresearch/spidr", "spidr_base")
        super().__init__("spidr", model, Wav2Vec2FeatureExtractor(), frame_rate=50)

    @torch.no_grad()
    def get_representations(self, audio, attention_mask=None):
        cnn = self.model.feature_extractor(audio)
        student = self.model.student
        with capture_inputs({"embeds": student.layers[0]}) as captured:
            # full layer outputs, as for the other models (the default is before the residual)
            layers = student.get_intermediate_outputs(
                self.model.feature_projection(cnn),
                attention_mask,
                before_residual=False,
            )
        return {
            "CNN": cnn,
            "embeds": captured["embeds"],
            **name_transformer_layers(layers),
        }
