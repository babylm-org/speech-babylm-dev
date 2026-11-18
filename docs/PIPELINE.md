### Setup
To set up the codebase, use your favorite package manager (uv, pixi, conda, ...).
The packages are listed in `requirements.txt`, and the project supports "Python>=3.12".
- With pip: `pip install -r requirements.txt`.
- with pixi `pixi install`.

To set up the evaluation, run `pixi run python -m scripts.eval.prepare_eval --models MODELS --tasks TASKS`, where models and tasks are separated by spaces.
The script downloads the evaluation data and model weights of the specified models and tasks.
Without `--models` or `--tasks`, it downloads the default models and tasks, defined in `src/speechlm/zero_shot/config.py`:

```python
DEFAULT_MODELS = [
    "facebook/wav2vec2-base",
    "microsoft/wavlm-base",
    "spidr",
    "facebook/hubert-base-ls960",
    "MarvinLvn/BabyHuBERT",
    "melhubert",
    "cpc",
]

DEFAULT_TASKS = [
    "triphone-dev-clean",
    "triphone-dev-other",
    "stress",
    "stress_syn",
    "stress_kokoro",
    "babyslm-lexical",
    "babyslm-syntactic",
]
```

### Evaluation
To evaluate models, run `pixi run python -m scripts.eval.run_eval --models MODELS --tasks TASKS` (with the same defaults). Optional arguments are:
- `--fast`: evaluate on a fixed subset of each task, with fewer examples and speakers. 
- `--layers`: score only these layers in ABX, e.g. `--layers T6 T12` (they must exist in all specified models).
- `--batch_size`: files per forward pass (default 16).
- `--keep_activations`: don't delete model activations after ABX evaluation (WARNING: these can take up a lot of space).

Evaluation data is stored in `data/`, model activations in `activations/` (deleted after evaluation, unless `--keep_activations` is given), and evaluation results in `results/<model>/<task>/`. Fast evaluation results are stored in a `<task>/fast/` subfolder.

The folder structure of the evaluation pipeline is given below.

```
speech-babylm-dev/
├── docs/           documentation
├── scripts/        entry points
├── src/speechlm/   source code
├── data/           evaluation data
├── models/         model weights (Hugging Face cache)
├── activations/    layer activations during ABX
└── results/        evaluation results
```

Important source code files are shown below.

```
src/speechlm/
└── zero_shot/                       evaluation code
    ├── config.py                    paths, default models and tasks
    ├── model_registry.py            all available models
    ├── task_registry.py             all available evaluations
    ├── models/
    │   ├── base.py                  SpeechLM: base interface
    │   ├── hf.py, cpc.py, ...       one file per model
    │   └── vendors/                 contains 3rd party implementations
    └── tasks/
        ├── base.py                  EvalTask: base interface
        ├── abx/
        │   ├── task.py              the tasks: data, setup and evaluation
        │   ├── conditions.py        ABX conditions (fastabx)
        │   ├── get_activations.py   save each layer's activations
        │   └── run_abx.py           score each layer
        └── babyslm/
            ├── task.py              the tasks: data, setup and evaluation
            └── evaluate.py          score minimal pairs
```

### Adding a new model

Adding a new model to the pipeline takes a few simple steps.

0. Check whether an implementation in `models/` already covers your model. In particular, `hf.py` covers Hugging Face models with a CNN encoder and transformer layers (e.g. Wav2Vec2, HuBERT, WavLM, ...).

1. Add `src/speechlm/zero_shot/models/my_model.py` with a subclass of `SpeechLM` (`models/base.py`):

```python
import torch

from speechlm.zero_shot.models.base import SpeechLM, name_transformer_layers


class MyModel(SpeechLM):
    def __init__(self, model_dir):
        model = ...         # load the weights, downloading them into model_dir
        preprocessor = ...  # e.g. Wav2Vec2FeatureExtractor()
        frame_rate = 50     # representation frames per second, needed for ABX
        super().__init__("my-model", model, preprocessor, frame_rate=frame_rate)

    # capture_inputs() in base.py captures activations that the model does not return
    # (e.g. input embeddings) during the forward pass
    @torch.no_grad()
    def get_representations(self, audio, attention_mask=None):
        # {layer name: (batch, frames, dim)}, from input to output: "CNN", "embeds", "T1", ...
        ...

    # only for speech language models
    def get_log_probs(self, audio, attention_mask=None):
        # the log probability of each utterance, of shape (batch,)
        ...
```

2. Add your model to `get_model` in `src/speechlm/zero_shot/model_registry.py`:

```python
        case "my-model":
            model = MyModel(model_dir)
```

3. (Optional) Add your model to `DEFAULT_MODELS` in `src/speechlm/zero_shot/config.py` to evaluate it by default.

### Adding a new task

0. Check if you can adapt one of the existing tasks:
   - ABX on new data: add a subclass of `ABXEval` in `tasks/abx/task.py`, like `TriphoneEval`, pointing to your audio and item file.
   - A new ABX contrast: add a subclass of `ABXCondition` in `tasks/abx/conditions.py`, setting what differs (`on`) and what is shared (`by`).
   - Minimal pairs scored by log probability: reuse `score_pairs` from `tasks/babyslm/evaluate.py`.

1. Add `src/speechlm/zero_shot/tasks/my_task/task.py` with a subclass of `EvalTask` (`tasks/base.py`):

```python
from speechlm.zero_shot.config import DATA_DIR
from speechlm.zero_shot.tasks.base import EvalTask


class MyTaskEval(EvalTask):
    def __init__(self):
        self.name = "my-task"

    def setup(self):
        # download and prepare the data under DATA_DIR / "my-task".
        ...
        self.make_fast()

    def make_fast(self):
        # write the fixed subset for --fast, if it does not exist
        ...

    def evaluate(self, model, batch_size=16, fast=False, **_):
        out_dir = self.results_dir(model, fast)  # results/<model>/my-task/
        if (out_dir / "results.csv").exists():
            return  # results are skipped, so a failed run can be restarted
        ...  # score the model, then write the results in out_dir
```

2. Add your task to `TASKS` in `src/speechlm/zero_shot/task_registry.py`:

```python
        MyTaskEval(),
```

3. (Optional) Add your task to `DEFAULT_TASKS` in `src/speechlm/zero_shot/config.py` to run it by default.
