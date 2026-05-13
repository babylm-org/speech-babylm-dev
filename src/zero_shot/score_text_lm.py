""" Score text LMs using minicons."""
from pathlib import Path

import pandas as pd
from minicons import scorer
import torch

def score_text_lm(task, split, model_name, backend):
    in_path = Path("data/babyslm/") / task / split / "gold.csv"
    stimuli = pd.read_csv(in_path)
    transcriptions = stimuli["transcription"].tolist()
    scorer_class = (
        scorer.IncrementalLMScorer if backend == "causal" else scorer.MaskedLMScorer
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = scorer_class(model_name, device=device)
    scores = model.sequence_score(transcriptions)
    out_path = Path("results") / Path(model_name).name / task / split
    out_path.mkdir(parents=True, exist_ok=True)
    stimuli["score"] = scores
    stimuli.to_csv(out_path / "scores.csv", index=False)
