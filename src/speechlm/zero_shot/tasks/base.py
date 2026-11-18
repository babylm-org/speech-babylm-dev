from abc import ABC, abstractmethod

from speechlm.zero_shot.config import RESULT_DIR
from speechlm.zero_shot.models.base import SpeechLM


class EvalTask(ABC):
    """A benchmark: its data, and how to score a model on it.
    name: the task's name in src/speechlm/zero_shot/task_registry.py
    """

    name: str

    @abstractmethod
    def setup(self):
        """Fetch and prepare the task's data. What exists is skipped."""

    @abstractmethod
    def evaluate(self, model: SpeechLM, batch_size=16, fast=False, **options):
        """Score a loaded model (model_registry.get_model) and save the results in
        self.results_dir(model, fast).

        Results that already exist are skipped, so an interrupted run can be restarted.

        fast: the fast evaluation, on the task's fixed sample, with its results kept apart.
        options: settings only some tasks have (see each task), the others ignore them.
        """

    def results_dir(self, model: SpeechLM, fast=False):
        """results/<model>/<task>/, or results/<model>/<task>/fast/ for the fast evaluation."""
        out_dir = RESULT_DIR / model.name / self.name
        return out_dir / "fast" if fast else out_dir
