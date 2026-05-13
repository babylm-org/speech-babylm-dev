from .compute_syntactic import evaluate_syntactic
from .score_text_lm import score_text_lm

score_text_lm(
    task="syntactic",
    split="dev",
    model_name="timinar/baby-llama-58m",
    backend="causal"
)
evaluate_syntactic(
    output_dir="results/baby-llama-58m/syntactic/dev/",
    gold_file="data/babyslm/syntactic/dev/gold.csv",
    submission_file="results/baby-llama-58m/syntactic/dev/scores.csv",
    is_text=True,
)