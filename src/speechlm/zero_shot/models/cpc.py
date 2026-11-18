import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import Wav2Vec2FeatureExtractor

from speechlm.zero_shot.config import CHECKPOINT_REPO
from speechlm.zero_shot.models.base import SpeechLM
from speechlm.zero_shot.models.vendors.cpc.model import CPCModel


class CPC(SpeechLM):
    """CPC (https://github.com/facebookresearch/CPC_audio): a CNN encoder and an LSTM.

    Representations: "CNN", "LSTM".
    """

    def __init__(self, model_dir):
        model = CPCModel()
        checkpoint = hf_hub_download(
            CHECKPOINT_REPO, "cpc/cpc_checkpoint_106.safetensors", cache_dir=model_dir
        )
        model.load_state_dict(load_file(checkpoint))
        preprocessor = Wav2Vec2FeatureExtractor(do_normalize=False)
        # 10 ms frames: the encoder strides multiply to 160 samples
        super().__init__("cpc", model, preprocessor, frame_rate=100)

    @torch.no_grad()
    def get_representations(self, audio, attention_mask=None):
        cnn = self.model.gEncoder(audio.unsqueeze(1)).transpose(1, 2)
        lstm, _ = self.model.gAR.baseNet(cnn)
        return {"CNN": cnn, "LSTM": lstm}
