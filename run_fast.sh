#!/usr/bin/env bash
# Fast evaluation: like run.sh, but downloads and evaluates only the fixed subset of each task.
# Models: src/speechlm/zero_shot/model_registry.py; tasks: src/speechlm/zero_shot/task_registry.py
set -euo pipefail

# everything printed also goes to logs/<script>_<date-time>.log
mkdir -p logs
exec > >(tee "logs/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log") 2>&1

MODELS=(
    facebook/wav2vec2-base
    microsoft/wavlm-base
    spidr
    facebook/hubert-base-ls960
    MarvinLvn/BabyHuBERT
    melhubert
    cpc
    # speech language models: the only ones scored on BabySLM
    slprl/slam
    slprl/slam_scaled
    slprl/slam_large
    SpeechLM-librispeech-k128
    SpeechLM-librispeech-k256
    SpeechLM-librispeech-k512
    SpeechLM-librispeech-k1024
)
TASKS=(
    triphone-dev-clean
    triphone-dev-other
    stress
    stress_syn
    stress_kokoro
    babyslm-lexical
    babyslm-syntactic
)
# ABX: score only these layers, by name (e.g. CNN T6) or position (0 the first, -1 the last)
# in each model's layers (see its get_representations); empty for all layers
LAYERS=()
# files per forward pass
BATCH_SIZE=32

pixi run python -m scripts.eval.prepare_eval --fast --models "${MODELS[@]}" --tasks "${TASKS[@]}"
pixi run python -m scripts.eval.run_eval --fast --models "${MODELS[@]}" --tasks "${TASKS[@]}" \
    --batch_size "$BATCH_SIZE" ${LAYERS[@]+--layers "${LAYERS[@]}"}
