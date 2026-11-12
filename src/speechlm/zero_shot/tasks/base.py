"""The interface of evaluation tasks, so any task can be run on any model the same way."""

from abc import ABC, abstractmethod

from speechlm.zero_shot.models.base import SpeechLM


class EvalTask(ABC):
    """A benchmark: its data, and how to score a model on it.

        name: the task's name in src/speechlm/zero_shot/task_registry.py
    """

    name: str

    @abstractmethod
    def download(self):
        """Fetch the task's data. Safe to re-run: what exists is skipped."""

    @abstractmethod
    def evaluate(self, model: SpeechLM, batch_size=16, fast=False, **options):
        """Score a loaded model (model_registry.get_model) and save the results, under
        results/<model.name>/.

        Results that already exist are skipped, so an interrupted run can be restarted.
        fast: the fast evaluation, on the task's fixed sample, with its results kept apart.
        options: settings only some tasks have (see each task); the others ignore them.
        """
