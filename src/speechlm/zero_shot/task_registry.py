"""The evaluation tasks by name, as model_registry.py does for models.


    task = get_task("babyslm-syntactic")
    task.download()
    model = get_model("slprl/slam").to("cuda")
    task.evaluate(model)
"""

from .tasks.abx.conditions import ProsodicABX, TriphoneABX
from .tasks.abx.task import ProsodicEval, TriphoneEval
from .tasks.babyslm.task import BabySLMEval
from .tasks.base import EvalTask

TASKS: dict[str, EvalTask] = {
    task.name: task
    for task in [
        *(TriphoneEval(split) for split in TriphoneABX.splits),
        *(ProsodicEval(name) for name in ProsodicABX.sets),
        BabySLMEval("lexical"),
        BabySLMEval("syntactic"),
    ]
}

DEFAULT_TASKS = [name for name, task in TASKS.items() if not isinstance(task, BabySLMEval)]


def get_task(name: str) -> EvalTask:
    if name not in TASKS:
        raise ValueError(f"Unsupported task: {name}; the tasks are {list(TASKS)}")
    return TASKS[name]
