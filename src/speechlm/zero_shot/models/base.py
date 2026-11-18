"""The interface of speech models, so the evaluations don't depend on how each model works.

Representations are named "CNN", "embeds" (the input to the transformer) and "T1", "T2", ...
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path

import torch


class SpeechLM(ABC):
    """A loaded speech model. Its constructor fetches whatever it is missing.

    name: the model's name in results, the last part of its name, e.g. "slam" for "slprl/slam".
    device: where the model is, set by to().
    """

    name: str
    device: str | torch.device = "cpu"

    def __init__(self, name: str, model: torch.nn.Module, preprocessor: Callable, frame_rate: int):
        """frame_rate: representation frames per second of audio."""
        self.name = Path(name).name
        self.model = model.eval()
        self.preprocessor = preprocessor
        self.frame_rate = frame_rate

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def get_preprocessor(self) -> Callable:
        """Return a callable with the interface of a Hugging Face feature extractor:
        preprocessor(signals, sampling_rate=16000, padding=True, return_tensors="pt").input_values
        """
        return self.preprocessor

    @abstractmethod
    def get_representations(
        self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Return {layer name: activations of shape (batch, frames, dim)}, from input to output.

        audio and attention_mask are the preprocessor's outputs, on the model's device.
        """

    def get_log_probs(
        self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return each utterance's log probability under the model, of shape (batch,).

        audio and attention_mask are the preprocessor's outputs, on the model's device.
        Only speech language models implement this.
        """
        raise NotImplementedError(f"{type(self).__name__} has no log probabilities")


@contextmanager
def capture_inputs(modules: dict[str, torch.nn.Module]):
    """Inside this context, forward passes save each module's first input under its name."""
    captured = {}

    def save_as(name):
        def hook(_module, inputs):
            captured[name] = inputs[0]

        return hook

    handles = [module.register_forward_pre_hook(save_as(name)) for name, module in modules.items()]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def name_transformer_layers(layers: Iterable[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"T{i}": activations for i, activations in enumerate(layers, start=1)}
