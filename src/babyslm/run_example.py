from src.babyslm.compute_syntactic import evaluate_syntactic
from src.babyslm.compute_lexical import evaluate_lexical   

evaluate_syntactic(
    output='example/babyberta/syntactic/dev_scores',
    gold='data/babyslm/',
    predicted='results/babyslm/babyberta/',
    kind='dev',
    is_text=True
)

evaluate_lexical(
    output='example/librivox_1024h/lexical/dev_scores',
    gold='data/babyslm/',
    predicted='results/babyslm/librivox_1024h',
    kind='dev',
)