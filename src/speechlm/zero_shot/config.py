"""Paths and the model list shared by the zero-shot tasks (paths are relative to the repo root)."""

from pathlib import Path

# saved model weights uploaded to HF
CHECKPOINT_REPO = "nikitastheo/speech-babylm-checkpoints"

MODEL_DIR = Path("models")

DATA_DIR = Path("data/eval")
ACTIVATION_DIR = Path("activations")
RESULT_DIR = Path("results")
# for ABX
ITEM_DIR = DATA_DIR / "items"

# the default models of scripts/eval/
MODELS = [
    "facebook/wav2vec2-base",
    "microsoft/wavlm-base",
    "spidr",
    "facebook/hubert-base-ls960",
    "MarvinLvn/BabyHuBERT",
    "melhubert",
    "cpc",
]


def results_dir(model_name_or_path, family):
    """Where a model's results for a task family go: results/<model>/<family>/, e.g.
    results/cpc/abx/."""
    return RESULT_DIR / Path(model_name_or_path).name / family
