#!/usr/bin/env bash
# Fast evaluation: like run.sh, but downloads and evaluates only the fixed subset of each task.
# Models: src/speechlm/zero_shot/model_registry.py; tasks: src/speechlm/zero_shot/task_registry.py
set -euo pipefail

MODELS=(
    facebook/wav2vec2-base
    microsoft/wavlm-base
    spidr
    facebook/hubert-base-ls960
    MarvinLvn/BabyHuBERT
    melhubert
    cpc
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
LAYERS=(0)

pixi run python -m scripts.eval.prepare_eval --fast --models "${MODELS[@]}" --tasks "${TASKS[@]}"
pixi run python -m scripts.eval.run_eval --fast --models "${MODELS[@]}" --tasks "${TASKS[@]}" \
    ${LAYERS[@]+--layers "${LAYERS[@]}"}
