set -e
pixi run python -m scripts.eval.prepare_eval\
    --models MarvinLvn/BabyHuBERT\
    --tasks triphone-dev-clean
pixi run python -m scripts.eval.run_eval\
    --models MarvinLvn/BabyHuBERT\
    --tasks triphone-dev-clean\
    --fast \
    --layers embeds
