### Speech Baby LM

The environment is managed with [pixi](https://pixi.sh): run `pixi install`. The code is the `speechlm` package in `src/speechlm/` (`dataset/` and `zero_shot/`), which pixi installs in editable mode, so `from speechlm.zero_shot.models import SpeechLM` works anywhere and edits apply without reinstalling.

#### Models

Every speech model implements the `SpeechLM` interface in `src/speechlm/zero_shot/models/base.py`. Each implementation has its own module in `src/speechlm/zero_shot/models/`: `HFModel` (`hf.py`) covers Hugging Face models such as Wav2Vec2, HuBERT and WavLM, and BabyHuBERT, SpidR, CPC, MelHuBERT and Slam have their own subclasses. To add a model, add a module with a `SpeechLM` subclass, have its constructor fetch the weights it is missing and implement `get_representations`, and add it to `get_model` in `src/speechlm/zero_shot/model_registry.py`. Speech language models also implement `get_log_probs`. Third-party model code lives in `src/speechlm/zero_shot/models/vendors/` with its licenses.

#### Tasks

Every evaluation implements the `EvalTask` interface in `src/speechlm/zero_shot/tasks/base.py` (`download` and `evaluate`), and `src/speechlm/zero_shot/task_registry.py` names them, as `model_registry.py` does for models. `evaluate` takes a model loaded with `get_model`, so `run_eval` loads each model once for all its tasks. `scripts/eval/` runs any task on any model:

```bash
python -m scripts.eval.prepare_eval  # weights -> models/, data -> data/eval/
python -m scripts.eval.run_eval      # results -> results/<model>/<abx|babyslm>/
python -m scripts.eval.run_example   # one model's representations of one file
```

`prepare_eval` and `run_eval` take `--models` (names in `model_registry.py`) and `--tasks` (names in `task_registry.py`), and the defaults are the ABX tasks. `run_eval --fast` is the fast evaluation: each task on a fixed sample (triphone: ~10% of the split; prosodic: the whole set, already small; BabySLM: 10% of the pairs), with its results saved apart from the full evaluation's. Checkpoints that are not on Hugging Face are mirrored in [nikitastheo/speech-babylm-checkpoints](https://huggingface.co/nikitastheo/speech-babylm-checkpoints). Gated or private Hugging Face models need `HF_TOKEN` in `.env`.

#### ABX

`src/speechlm/zero_shot/tasks/abx/` (task classes in `task.py`) calculates ABX, layer by layer, with [fastabx](https://github.com/bootphon/fastabx) ([arXiv](https://arxiv.org/abs/2505.02692)). The tasks are the ZeroSpeech 2021 triphone ABX (`triphone-dev-clean`, `triphone-dev-other`; with `--fast`, 10 speakers x 25 recordings of each, results in `results/<model>/abx/triphone-<split>-sample/`) and [Prosodic ABX](https://arxiv.org/abs/2604.02102) (`stress`, `stress_syn`, `stress_kokoro`). For example: `run_eval --models cpc --tasks stress`.

#### BabySLM

`src/speechlm/zero_shot/tasks/babyslm/` (task class in `task.py`) evaluates speech language models (those with `get_log_probs`) on [BabySLM](https://arxiv.org/abs/2306.01506)'s spoken minimal pairs: `babyslm-lexical` (sWUGGY-like) and `babyslm-syntactic` (sBLIMP-like). `prepare.py` downloads them and lists each pair, its correct and wrong files side by side, in `data/eval/babyslm/<task>/dev/pairs.csv`, and a fixed 10% of them (of each type, for syntactic) in `pairs-sample.csv` for `--fast`. `evaluate.py` scores the pairs as [slamkit](https://github.com/slp-rl/slamkit) scores sWUGGY and sBLIMP: a pair is right when the correct file has the higher log probability, and a tie counts half. The only speech language model so far is [Slam](https://arxiv.org/abs/2502.15814) (`slprl/slam`, `slprl/slam_scaled`, `slprl/slam_large`), run with slamkit.

```bash
python -m scripts.eval.run_eval --models slprl/slam --tasks babyslm-lexical babyslm-syntactic
# -> results/slam/babyslm/<task>_dev.csv (each pair with its log probabilities and score)
#    and <task>_dev.json (accuracy, and by type of syntactic pair); with --fast, <task>_dev-sample.*
```
