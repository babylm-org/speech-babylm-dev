"""Score BabySLM's minimal pairs with a speech language model, as slamkit scores sWUGGY and
sBLIMP (slamkit/metric/modelling_metric.py): a pair is right when its correct file has the
higher log probability, and a tie counts half.
"""

import numpy as np
import pandas as pd
import soundfile as sf
import tqdm


def log_probs(model, wav_dir, filenames, batch_size=16):
    """The model's log probability of each wav_dir/<filename>.wav."""
    preprocessor = model.get_preprocessor()
    scores = []
    for start in tqdm.trange(0, len(filenames), batch_size, desc="Scoring files"):
        batch = filenames[start : start + batch_size]
        signals, rates = zip(*(sf.read(wav_dir / f"{f}.wav") for f in batch), strict=True)
        inputs = preprocessor(
            list(signals), sampling_rate=rates[0], padding=True, return_tensors="pt"
        ).to(model.device)
        scores += model.get_log_probs(
            inputs.input_values, inputs.get("attention_mask")
        ).tolist()
    return scores


def score_pairs(model, pairs, wav_dir, batch_size=16) -> pd.DataFrame:
    """pairs (a pairs.csv) with each file's log probability and the pair's score: 1, 0.5 or 0."""
    filenames = list(pd.unique(pairs[["filename_correct", "filename_wrong"]].values.ravel()))
    by_file = dict(
        zip(filenames, log_probs(model, wav_dir, filenames, batch_size), strict=True)
    )
    pairs = pairs.assign(
        log_prob_correct=pairs["filename_correct"].map(by_file),
        log_prob_wrong=pairs["filename_wrong"].map(by_file),
    )
    difference = pairs["log_prob_correct"] - pairs["log_prob_wrong"]
    return pairs.assign(score=(np.sign(difference) + 1) / 2)
