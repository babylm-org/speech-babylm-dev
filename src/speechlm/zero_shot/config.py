from pathlib import Path

# saved model weights uploaded to HF
# TODO: move this to BabyLM-repo
CHECKPOINT_REPO = "nikitastheo/speech-babylm-checkpoints"

# paths relative to root repo
MODEL_DIR = Path("models")
DATA_DIR = Path("data/eval")
ACTIVATION_DIR = Path("activations")
RESULT_DIR = Path("results")
ABX_ITEM_DIR = DATA_DIR / "items"

# the default models of scripts/eval/
DEFAULT_MODELS = [
    "facebook/wav2vec2-base",
    "microsoft/wavlm-base",
    "spidr",
    "facebook/hubert-base-ls960",
    "MarvinLvn/BabyHuBERT",
    "melhubert",
    "cpc",
]

# the default tasks of scripts/eval/, names from task_registry.py
DEFAULT_TASKS = [
    "triphone-dev-clean",
    "triphone-dev-other",
    "stress",
    "stress_syn",
    "stress_kokoro",
    "babyslm-lexical",
    "babyslm-syntactic",
]
