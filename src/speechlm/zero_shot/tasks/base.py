import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

from huggingface_hub import snapshot_download

from speechlm.zero_shot.config import EVAL_DATA_DIR, EVAL_REPO, RESULT_DIR
from speechlm.zero_shot.models.base import SpeechLM


class EvalTask(ABC):
    """A benchmark: its data, and how to score a model on it.

    name: the task's name in src/speechlm/zero_shot/task_registry.py
    family: its folder in EVAL_REPO, e.g. "triphone"
    subset: its files in that folder: <subset>.zip (the audio) and its data file, e.g.
        "dev-clean" for triphone/dev-clean.zip and triphone/dev-clean.item

    Its data is in EVAL_REPO (built by scripts/eval/build_eval_data.py):
        evaluation_data/full_eval/<family>/<subset>.*       the whole set
        evaluation_data/fast_eval/<family>_fast/<subset>.*  the fixed subset of --fast
    """

    name: str
    family: str
    subset: str

    def setup(self, fast_only=False):
        """Download the task's full and fast data from EVAL_REPO, and unzip the audio. What
        exists is skipped.

        fast_only: only the fast data, a small download.
        """
        for fast in (True,) if fast_only else (False, True):
            folder = self.data_dir(fast)
            print(f"==> {self.name}: {folder}")
            in_repo = folder.relative_to(EVAL_DATA_DIR.parent).as_posix()  # evaluation_data/...
            snapshot_download(
                EVAL_REPO,
                repo_type="dataset",
                local_dir=EVAL_DATA_DIR.parent,
                allow_patterns=[f"{in_repo}/{self.subset}.*"],
            )
            audio_dir = self.audio_dir(fast)
            if not audio_dir.is_dir():
                # via a .part folder, so an interrupted unzip is not mistaken for a finished one
                part = audio_dir.with_name(audio_dir.name + ".part")
                with zipfile.ZipFile(folder / f"{self.subset}.zip") as zf:
                    zf.extractall(part)
                part.rename(audio_dir)

    def data_dir(self, fast=False) -> Path:
        """The folder of the task's data: full_eval/<family>/ or fast_eval/<family>_fast/."""
        if fast:
            return EVAL_DATA_DIR / "fast_eval" / f"{self.family}_fast"
        return EVAL_DATA_DIR / "full_eval" / self.family

    def audio_dir(self, fast=False) -> Path:
        """The task's unzipped audio, one file per recording."""
        return self.data_dir(fast) / self.subset

    @abstractmethod
    def evaluate(self, model: SpeechLM, batch_size=16, fast=False, **options):
        """Score a loaded model (model_registry.get_model) and save the results in
        self.results_dir(model, fast).

        Results that already exist are skipped, so an interrupted run can be restarted.

        fast: the fast evaluation, on the task's fixed subset, with its results kept apart.
        options: settings only some tasks have (see each task), the others ignore them.
        """

    def results_dir(self, model: SpeechLM, fast=False):
        """results/full/<model>/<task>/, or results/fast/<model>/<task>/ for the fast evaluation."""
        return RESULT_DIR / ("fast" if fast else "full") / model.name / self.name
