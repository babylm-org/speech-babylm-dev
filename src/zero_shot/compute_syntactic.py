"""Adapted from ZeroSpeech 2021"""

import pathlib
import pandas


def load_data(gold_file, submission_file, is_text=False):
    gold_file = pathlib.Path(gold_file)
    submission_file = pathlib.Path(submission_file)
    if not gold_file.is_file():
        raise ValueError(f"file not found: {gold_file}")
    if not submission_file.is_file():
        raise ValueError(f"file not found: {submission_file}")

    # load them as data frames indexed by filenames
    gold = pandas.read_csv(gold_file)

    score = pandas.read_csv(submission_file)[["id", "filename", "score", "voice"]]
    # voice doesn't matter for stext LMs
    # just get a random voic
    if is_text:
        voices = gold["voice"].unique()
        gold = gold[gold["voice"] == voices[0]]
        score = score[score["voice"] == voices[0]]

    gold['merge_id'] = gold['id'].astype(str) + "_" + gold['filename']
    score['merge_id'] = score['id'].astype(str) + "_" + score['filename']
    score.drop(columns=["filename", "voice", "id"], inplace=True)
    # laboriously create pairs of correct and incorrect sentences
    data = pandas.merge(gold, score, on="merge_id", how="inner")
    print(data)
    data = data.reset_index(drop=True)
    correct = data.loc[data["correct"] == 1].reset_index().rename(lambda x: "s_" + x, axis=1)
    wrong = data.loc[data["correct"] == 0].reset_index().rename(lambda x: "ns_" + x, axis=1)
    data = pandas.concat(
        [
            correct,
            wrong,
        ],
        axis=1,
    )
    assert (data["ns_id"] == data["s_id"]).all(), (
        "Mismatch between sentence and non sentence ids."
    )
    assert (data["ns_voice"] == data["s_voice"]).all(), (
        "Mismatch between sentence and non sentence voices."
    )
    data.drop(
        [
            "s_index",
            "ns_index",
            "ns_voice",
            "ns_type",
            "ns_subtype",
            "s_correct",
            "ns_correct",
            "ns_id",
        ],
        axis=1,
        inplace=True,
    )
    data.rename(
        {
            "s_id": "id",
            "s_voice": "voice",
            "s_type": "type",
            "s_subtype": "subtype",
            "s_transcription": "sentence",
            "ns_transcription": "non sentence",
            "s_score": "score sentence",
            "ns_score": "score non sentence",
        },
        axis=1,
        inplace=True,
    )
    if data[["score sentence", "score non sentence"]].isna().sum().sum():
        print(data[data["score sentence"].isna()])
        print(data[data["score non sentence"].isna()])
        raise ValueError("Found some NaN in the predicted scores. Aborting.")
    return data


def evaluate_all(data):
    score = data.loc[:, ["score sentence", "score non sentence"]].to_numpy()
    data["score"] = 0.5 * (score[:, 0] == score[:, 1]) + (score[:, 0] > score[:, 1])
    return data.copy()


def evaluate_by_pair(data):
    """Returns a data frame with the computed scores by (grammatical sentence, ungrammatical sentence) pair

    Parameters
    ----------
    data : pandas.DataFrame
        The result of `load_data`

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated (sentence, non sentence) pairs, the data frame has the
        columns: 'sentence', 'non sentence' 'type' and 'score'.

    """
    # compute the score for each pair in an additional 'score' column, then
    # delete the 'score sentence' and 'score non sentence' columns that become useless
    score = data.loc[:, ["score sentence", "score non sentence"]].to_numpy()
    data["score"] = 0.5 * (score[:, 0] == score[:, 1]) + (score[:, 0] > score[:, 1])
    data.drop(columns=["score sentence", "score non sentence"], inplace=True)
    score = data.groupby(["type", "subtype", "id"]).apply(
        lambda x: (
            x.name[0],  # type (from index)
            x.name[1],  # subtype (from index)
            x["sentence"].iat[0],  # sentence
            x["non sentence"].iat[0],  # non sentence
            x["score"].mean(),
        ),
    )
    score = pandas.DataFrame(
        score.to_list(),
        columns=["type", "subtype", "sentence", "non sentence", "score"],
    )
    return score


def evaluate_by_type(by_pair):
    """Returns a data frame with mean scores by syntax error type

    Parameters
    ----------
    by_pair: pandas.DataFrame
        The output of `evaluate_by_pair`

    Returns
    -------
    by_type : pandas.DataFrame
        The score collapsed on types, the data frame has the
        following columns: 'type', 'score'.

    """
    data = (
        by_pair.score.groupby([by_pair["type"]])
        .agg(n="count", score="mean", std="std")
        .reset_index()
    )
    return data


def evaluate(gold_file, submission_file, is_text=False):
    """Returns the score by sentences pair and by syntax type

    Parameters
    ----------
    gold_file : path
        The gold file (csv format) for the lexical dataset (test or dev).
    submission_file : path
        The submission corresponding to the provided gold file.

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated pairs, the data frame has the columns:
        'sentence', 'non sentence' and 'score'.
    by_type : pandas.DataFrame
        The score collapsed on syntax errors types, the data frame has the
        following columns: 'type', 'score'.

    Raise
    -----
    ValueError
        If the input files cannot be opened or in case of data mismatch between
        the two files.

    """
    data = load_data(gold_file, submission_file, is_text)

    all_trials = evaluate_all(data)
    by_pair = evaluate_by_pair(data)
    by_type = evaluate_by_type(by_pair)
    by_pair.drop(["type", "subtype"], axis=1, inplace=True)

    return all_trials, by_pair, by_type


def write_csv(frame, filename):
    frame.to_csv(filename, index=False, float_format="%.4f")
    print(f"  > Wrote {filename}")


def write_final(acc, filename):
    with open(filename, "w") as fin:
        print(acc, file=fin)
    print(f"  > Wrote {filename}")


def evaluate_syntactic(output_dir, gold_file, submission_file, is_text=False):
    gold_file = pathlib.Path(gold_file)
    submission_file = pathlib.Path(submission_file)
    output = pathlib.Path(output_dir)

    print("Evaluating syntactic...")
    all_trials, by_pair, by_type = evaluate(gold_file, submission_file, is_text=is_text)

    output.mkdir(exist_ok=True, parents=True)
    write_csv(all_trials, output / "score_syntactic_all_trials.csv")
    write_csv(by_pair, output / "score_syntactic_by_pair.csv")
    write_csv(by_type, output / "score_syntactic_by_type.csv")

    # write final score
    write_final(by_pair["score"].mean(), output / "overall_accuracy_syntactic.txt")
    print(by_pair["score"].mean())