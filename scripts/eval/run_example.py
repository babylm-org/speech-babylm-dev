"""Load a model by name (src/speechlm/zero_shot/model_registry.py) and print the shape of each of its
representations of one BabySLM file.

Usage:
    python -m scripts.eval.run_example
"""

import soundfile as sf
import torch

from speechlm.zero_shot.config import MODEL_DIR
from speechlm.zero_shot.model_registry import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# https://arxiv.org/pdf/2006.11477
model = get_model("facebook/wav2vec2-base", MODEL_DIR).to(device)
preprocessor = model.get_preprocessor()

example_audio, sampling_rate = sf.read(
    "data/eval/babyslm/lexical/dev/wavs/aaron's-en-US-Wavenet-B.wav"
)
inputs = preprocessor(
    example_audio, sampling_rate=sampling_rate, return_tensors="pt"
).input_values.to(device)
representations = model.get_representations(inputs)
print({layer: tuple(acts.shape) for layer, acts in representations.items()})
