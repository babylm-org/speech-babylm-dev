from pathlib import Path

# mirrors of third-party model weights (CPC, MelHuBERT, Slam's k-means)
CHECKPOINT_REPO = "BabyLM-community/Speech-BabyLM-Checkpoints"

# the evaluation data (full_eval/ and fast_eval/), built by scripts/eval/build_eval_data.py
EVAL_REPO = "BabyLM-community/Speech-BabyLM-Evals"

# paths relative to root repo
MODEL_DIR = Path("models")
EVAL_DATA_DIR = Path("data/evaluation_data")  # EVAL_REPO's evaluation_data/, audio unzipped
RAW_DATA_DIR = Path("data/raw")  # the original sources, only for build_eval_data.py
ACTIVATION_DIR = Path("activations")
RESULT_DIR = Path("results")

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
