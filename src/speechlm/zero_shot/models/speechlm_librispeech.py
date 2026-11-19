"""SpeechLM (https://github.com/mynlp/speechLM) trained on LibriSpeech: an OPT language model
over k-means units of HuBERT BASE's layer 9 (20 ms frames, repeats merged). One model per
number of units K, on branch k<K> of BabyLM-community/SpeechLM-librispeech.
"""

from pathlib import Path

import torch
import torchaudio
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Wav2Vec2FeatureExtractor

from speechlm.zero_shot.models.babyhubert import TorchaudioHuBERT

REPO = "BabyLM-community/SpeechLM-librispeech"
UNIT_LAYER = 9


class SpeechLMLibrispeech(TorchaudioHuBERT):
    """get_log_probs is the mean log probability per unit, as in the model's README.
    Representations are those of the speech encoder, HuBERT BASE, the same for every K.
    """

    def __init__(self, k, model_dir):
        lm_dir = Path(snapshot_download(REPO, revision=f"k{k}", cache_dir=model_dir))
        self.centers = load_file(lm_dir / "kmeans.safetensors")["cluster_centers"]
        self.tokenizer = AutoTokenizer.from_pretrained(lm_dir)
        self.lm = AutoModelForCausalLM.from_pretrained(lm_dir).eval()
        # torchaudio's HuBERT BASE (fairseq's, LibriSpeech 960h), on which the k-means was fit
        encoder = torchaudio.pipelines.HUBERT_BASE.get_model(dl_kwargs={"model_dir": model_dir})
        # raw waveforms, as the model's README reads them; the attention mask gives the lengths
        preprocessor = Wav2Vec2FeatureExtractor(do_normalize=False, return_attention_mask=True)
        super().__init__(f"SpeechLM-librispeech-k{k}", encoder, preprocessor, frame_rate=50)

    def to(self, device):
        self.lm.to(device)
        self.centers = self.centers.to(device)
        return super().to(device)

    @torch.no_grad()
    def get_log_probs(self, audio, attention_mask=None):
        """Each utterance is encoded on its own: HuBERT's group norm would also normalize the
        batch padding."""
        if attention_mask is None:
            lengths = [audio.shape[1]] * len(audio)
        else:
            lengths = attention_mask.sum(dim=1).tolist()
        texts = []
        for signal, length in zip(audio, lengths, strict=True):
            layers, _ = self.model.extract_features(signal[None, :length], num_layers=UNIT_LAYER)
            # the nearest center, in the centers' float64, as scikit-learn's predict
            units = torch.cdist(layers[-1][0].double(), self.centers).argmin(dim=1)
            texts.append(" ".join(map(str, torch.unique_consecutive(units).tolist())))
        # the tokenizer appends <eos>; every token after the first is predicted
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        logits = self.lm(**inputs).logits[:, :-1].float()
        targets = inputs["input_ids"][:, 1:]
        mask = inputs["attention_mask"][:, 1:]
        log_probs = logits.log_softmax(dim=-1).gather(-1, targets[..., None])[..., 0]
        return (log_probs * mask).sum(dim=1) / mask.sum(dim=1)
