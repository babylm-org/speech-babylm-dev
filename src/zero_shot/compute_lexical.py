"""Adapted from ZeroSpeech 2021"""

import argparse
import pathlib
import sys

import pandas


def load_data(gold_file, submission_file, is_text=False):
    """Returns the data required for evaluation as a pandas data frame

    Each line of the returned data frame contains a pair (word, non word) and
    has the following columns: 'id', 'voice', 'frequency', 'word', 'score
    word', 'non word', 'score non word'.

    Parameters
    ----------
    gold_file : path
        The gold file for the lexical dataset (test or dev).
    submission_file : path
        The submission corresponding to the provided gold file.

    Returns
    -------
    data : pandas.DataFrame
        The data ready for evaluation

    Raise
    -----
    ValueError
        If the input files cannot be opened or in case of data mismatch between
        the two files.

    """
    # ensures the two input files are here
    for input_file in (gold_file, submission_file):
        if not pathlib.Path(input_file).is_file():
            raise ValueError(f"file not found: {input_file}")

    # load them as data frames indexed by filenames
    gold = pandas.read_csv(gold_file, header=0, index_col="filename").astype(
        {"frequency": pandas.Int64Dtype()}
    )
    score = pandas.read_csv(
        submission_file,
        sep=" ",
        header=None,
        names=["filename", "score"],
        usecols=[0, 1],
        index_col="filename",
    )

    if is_text:
        voices = gold["voice"].unique()
        gold = gold[gold["voice"] == voices[0]]
        # same number of lines is expected
        data = pandas.concat([gold, score], axis=1)
    else:
        # merge by filename
        data = pandas.merge(gold, score, on="filename")

    data.reset_index(drop=True, inplace=True)

    if len(data) != len(gold):
        raise ValueError("len(data) should be equal to len(gold). Aborting.")

    # if all non words have their textual version set to NaN, we take their phonemic version instead.
    if data[data.correct == 0]["word"].isnull().sum() == len(data[data.correct == 0]):
        data["word"] = data["phones"]
    data.drop(columns=["phones"], inplace=True)

    # going from a word per line to a pair (word, non word) per line
    words = (
        data.loc[data["correct"] == 1].reset_index().rename(lambda x: "w_" + x, axis=1)
    )
    non_words = (
        data.loc[data["correct"] == 0].reset_index().rename(lambda x: "nw_" + x, axis=1)
    )
    data = pandas.merge(
        words, non_words, left_on=["w_voice", "w_id"], right_on=["nw_voice", "nw_id"]
    )

    data.drop(
        [
            "w_index",
            "nw_index",
            "nw_voice",
            "nw_frequency",
            "w_correct",
            "nw_correct",
            "nw_id",
            "nw_length",
        ],
        axis=1,
        inplace=True,
    )
    data.rename(
        {
            "w_id": "id",
            "w_voice": "voice",
            "w_frequency": "frequency",
            "w_word": "word",
            "nw_word": "non word",
            "w_length": "length",
            "w_score": "score word",
            "nw_score": "score non word",
        },
        axis=1,
        inplace=True,
    )
    if data[["score word", "score non word"]].isna().sum().sum():
        raise ValueError("Found some NaN in the predicted scores. Aborting.")
    return data


def evaluate_all(data):
    score = data.loc[:, ["score word", "score non word"]].to_numpy()
    data["score"] = 0.5 * (score[:, 0] == score[:, 1]) + (score[:, 0] > score[:, 1])
    return data.copy()


def evaluate_by_pair(data):
    """Returns a data frame with the computed scores by (word, non word) pair

    Parameters
    ----------
    data : pandas.DataFrame
        The result of `load_data`

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated (word, non word) pairs, the data frame has the columns:
        'word', 'non word' 'frequency', 'length' and 'score'.

    """
    # compute the score for each pair in an additional 'score' column, then
    # delete the 'score word' and 'score non word' columns that become useless
    score = data.loc[:, ["score word", "score non word"]].to_numpy()
    data["score"] = 0.5 * (score[:, 0] == score[:, 1]) + (score[:, 0] > score[:, 1])
    data.drop(columns=["score word", "score non word"], inplace=True)

    # We average across non-words
    score = data.groupby(["voice", "word"]).apply(
        lambda x: (
            x.iat[0, 1],  # voice
            x.iat[0, 3],  # word
            "/".join(x["non word"]),  # non words
            x.iat[0, 2],  # frequency
            x.iat[0, 4],  # length
            x["score"].mean(),
        )
    )
    score = pandas.DataFrame(
        score.to_list(),
        columns=["voice", "word", "non words", "frequency", "length", "score"],
    )

    # We average across voices
    score = score.groupby("word").apply(
        lambda x: (
            x.iat[0, 1],  # word
            x.iat[0, 2],  # non words
            x.iat[0, 3],  # frequency
            x.iat[0, 4],  # length
            "/".join(x["voice"].astype(str)),  # voices
            x["score"].mean(),
        )
    )
    out = pandas.DataFrame(
        score.to_list(),
        columns=["word", "non words", "frequency", "length", "voices", "score"],
    )
    return out


def evaluate_by_frequency(by_pair):
    """Returns a data frame with mean scores by frequency bands

    The frequency is defined as the number of occurences of the word in the
    LibriSpeech dataset. The following frequency bands are considered : oov,
    1-5, 6-20, 21-100 and >100.

    Parameters
    ----------
    by_pair: pandas.DataFrame
        The output of `evaluate_by_pair`

    Returns
    -------
    by_frequency : pandas.DataFrame
        The score collapsed on frequency bands, the data frame has the
        following columns: 'frequency', 'score'.

    """
    bands = pandas.cut(
        by_pair.frequency,
        [0, 1, 5, 20, 100, float("inf")],
        labels=["oov", "1-5", "6-20", "21-100", ">100"],
        right=False,
    )

    return (
        by_pair.score.groupby(bands)
        .agg(n="count", score="mean", std="std")
        .reset_index()
    )


def evaluate_by_length(by_pair):
    """Returns a data frame with mean scores by word length

    Parameters
    ----------
    by_pair: pandas.DataFrame
        The output of `evaluate_by_pair`

    Returns
    -------
    by_length : pandas.DataFrame
        The score collapsed on word length, the data frame has the
        following columns: 'length', 'score'.

    """
    return (
        by_pair.score.groupby(by_pair.length)
        .agg(n="count", score="mean", std="std")
        .reset_index()
    )


def evaluate(gold_file, submission_file, is_text=False):
    """Returns the score by (word, non word) pair, by frequency and by length

    Parameters
    ----------
    gold_file : path
        The gold file (csv format) for the lexical dataset (test or dev).
    submission_file : path
        The submission corresponding to the provided gold file.

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated (word, non word) pairs, the data frame has the columns:
        'word', 'non word' and 'score'.
    by_frequency : pandas.DataFrame
        The score collapsed on frequency bands, the data frame has the
        following columns: 'frequency', 'score'.
    by_length : pandas.DataFrame
        The score collapsed on word length (in number of phones), the data
        frame has the following columns: 'length', 'score'.

    Raise
    -----
    ValueError
        If the input files cannot be opened or in case of data mismatch between
        the two files.

    """
    data = load_data(gold_file, submission_file, is_text)

    all_trials = evaluate_all(data)
    by_pair = evaluate_by_pair(data)
    by_frequency = evaluate_by_frequency(by_pair)
    by_length = evaluate_by_length(by_pair)
    by_pair.drop(["frequency", "length"], axis=1, inplace=True)

    return all_trials, by_pair, by_frequency, by_length


def write_csv(frame, filename):
    frame.to_csv(filename, index=False, float_format="%.4f")
    print(f"  > Wrote {filename}")


def write_final(acc, filename):
    with open(filename, "w") as fin:
        print(acc, file=fin)
    print(f"  > Wrote {filename}")


def evaluate_lexical(gold_file, submission_file, output_dir, is_text=False):
    gold_file = pathlib.Path(gold_file)
    submission_file = pathlib.Path(submission_file)
    output = pathlib.Path(output_dir)

    print(f"Evaluating lexical...")

    all_trials, by_pair, by_frequency, by_length = evaluate(
        gold_file, submission_file, is_text=is_text
    )

    output.mkdir(exist_ok=True, parents=True)
    write_csv(all_trials, output / f"score_lexical_all_trials.csv")
    write_csv(by_pair, output / f"score_lexical_by_pair.csv")
    write_csv(by_frequency, output / f"score_lexical_by_frequency.csv")
    write_csv(by_length, output / f"score_lexical_by_length.csv")

    # write final score
    write_final(
        by_pair["score"].mean(), output / f"overall_accuracy_lexical.txt"
    )
    print(by_pair["score"].mean())