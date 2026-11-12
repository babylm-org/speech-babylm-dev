from pathlib import Path

from .models.babyhubert import BabyHuBERT
from .models.base import SpeechLM
from .models.cpc import CPC
from .models.hf import HFModel
from .models.melhubert import MelHuBERT
from .models.slam import Slam
from .models.spidr import SpidR


def get_model(model_name_or_path: str, model_dir="models") -> SpeechLM:
    """Load a model by name, fetching what it is missing into model_dir.

    Its results are named after the last part of the name, e.g. "slam" for "slprl/slam".
    """
    match model_name_or_path:
        case (
            "facebook/wav2vec2-base"
            | "facebook/hubert-base-ls960"
            | "microsoft/wavlm-base"
        ):
            model = HFModel(model_name_or_path, cache_dir=model_dir)
        case "MarvinLvn/BabyHuBERT":
            model = BabyHuBERT(model_dir)
        case "spidr":
            model = SpidR()
        case "melhubert":
            model = MelHuBERT(model_dir)
        case "cpc":
            model = CPC(model_dir)
        case "slprl/slam" | "slprl/slam_scaled" | "slprl/slam_large":
            model = Slam(model_name_or_path, model_dir)
        case _:
            raise ValueError(f"Unsupported model: {model_name_or_path}")
    model.name = Path(model_name_or_path).name
    return model
