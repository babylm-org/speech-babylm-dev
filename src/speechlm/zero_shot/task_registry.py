from .tasks.abx.task import PROSODIC_SETS, TRIPHONE_SPLITS, ProsodicEval, TriphoneEval
from .tasks.babyslm.task import BabySLMEval
from .tasks.base import EvalTask

TASKS: dict[str, EvalTask] = {
    task.name: task
    for task in [
        *(TriphoneEval(split) for split in TRIPHONE_SPLITS),
        *(ProsodicEval(name) for name in PROSODIC_SETS),
        BabySLMEval("lexical"),
        BabySLMEval("syntactic"),
    ]
}


def get_task(name: str) -> EvalTask:
    if name not in TASKS:
        msg = f"Unsupported task: {name}; the tasks are {list(TASKS)}"
        raise ValueError(msg)
    return TASKS[name]
