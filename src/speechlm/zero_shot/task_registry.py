from .tasks.abx.task import ProsodicEval, TriphoneEval
from .tasks.babyslm.task import BabySLMEval
from .tasks.base import EvalTask

TASKS: dict[str, EvalTask] = {
    task.name: task
    for task in [
        TriphoneEval("dev-clean"),
        TriphoneEval("dev-other"),
        ProsodicEval("stress"),
        ProsodicEval("stress_syn"),
        ProsodicEval("stress_kokoro"),
        BabySLMEval("lexical"),
        BabySLMEval("syntactic"),
    ]
}


def get_task(name: str) -> EvalTask:
    if name not in TASKS:
        msg = f"Unsupported task: {name}; the tasks are {list(TASKS)}"
        raise ValueError(msg)
    return TASKS[name]
