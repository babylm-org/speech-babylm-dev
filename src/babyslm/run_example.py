from src.babyslm.babyberta.extract_prob_babyberta import score_babyberta
from src.babyslm.metrics.compute_syntactic import evaluate_syntactic

# input_path = "data/babyslm/lexical/"
# score_babyberta(input_path=input_path, task="lexical")

input_path = "data/babyslm/syntactic/"
score_babyberta(input_path=input_path, task="syntactic", model="babyberta1")

evaluate_syntactic(
    output="results/babyslm/babyberta/dev_scores",
    gold="data/babyslm/",
    predicted="results/babyslm/babyberta/",
    kind="dev",
    is_text=True,
)

# evaluate_lexical(
#     output="example/babyberta/lexical/dev_scores",
#     gold="data/babyslm/",
#     predicted="results/babyslm/babyberta/",
#     kind="dev",
#     is_text=True,
# )
