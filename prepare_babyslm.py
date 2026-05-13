import json
import os
from pathlib import Path

import pandas as pd


def download_data():
    Path("data/babyslm/lexical").mkdir(parents=True, exist_ok=True)
    Path("data/babyslm/syntactic").mkdir(parents=True, exist_ok=True)

    # Download dev
    os.system(
        "wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/dev.zip -P data/babyslm/lexical"
    )
    os.system(
        "wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/dev.zip -P data/babyslm/syntactic"
    )
    os.system("unzip data/babyslm/lexical/dev.zip -d data/babyslm/lexical")
    os.system("unzip data/babyslm/syntactic/dev.zip -d data/babyslm/syntactic")
    # remove zip
    os.system("rm data/babyslm/lexical/dev.zip")
    os.system("rm data/babyslm/syntactic/dev.zip")
    # fix name
    os.system("mv data/babyslm/syntactic/dev_16 data/babyslm/syntactic/dev")

    # download test (might be long)
    # os.system("wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/test.zip -P <DATA_LOCATION>/babyslm/lexical")
    # os.system("wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/test.zip -P <DATA_LOCATION>/babyslm/syntactic")

    # os.system("unzip <DATA_LOCATION>/babyslm/lexical/test.zip -d <DATA_LOCATION>/babyslm/lexical")
    # os.system("unzip <DATA_LOCATION>/babyslm/syntactic/test.zip -d <DATA_LOCATION>/babyslm/syntactic")


def format_data():

    for task in ["syntactic", "lexical"]:
        path = Path("data/babyslm") / task / "dev"

        (path / "wavs").mkdir(exist_ok=True)
        os.system(f"mv {path}/*.wav {path}/wavs")
        df = pd.read_csv(path / "gold.csv")
        samples = []
        for sample_id, sample_row in df.groupby("id"):
            per_voice = sample_row.groupby("voice")

            for _, pair in per_voice:
                row_correct = pair[pair["correct"] == 1]
                row_wrong = pair[pair["correct"] == 0]
                if task == "syntactic":
                    voice = row_correct["voice"].iloc[0]
                    sample = {
                        "id": str(sample_id) + f"_{voice}",
                        "UID": row_correct["type"].iloc[0],
                        "subtype": row_correct["subtype"].iloc[0],
                        "sentences": [
                            row_correct["transcription"].iloc[0],
                            row_wrong["transcription"].iloc[0],
                        ],
                        "label": 0,
                        "example_id" : sample_id,
                        "voice": voice,
                        "filenames": [
                            row_correct["filename"].iloc[0],
                            row_wrong["filename"].iloc[0],
                        ],
                    }
                else:
                    voice = row_correct["voice"].iloc[0].replace("en-US-Wavenet-", "")
                    sample = {
                        "id": str(sample_id) + f"_{voice}",
                        "UID": "lexical",
                        "word": row_correct["word"].iloc[0],
                        "phones": [
                            row_correct["phones"].iloc[0],
                            row_wrong["phones"].iloc[0],
                        ],
                        "label": 0,
                        "example_id" : sample_id,
                        "voice": voice,
                        "filenames": [
                            row_correct["filename"].iloc[0],
                            row_wrong["filename"].iloc[0],
                        ],
                    }

                samples.append(sample)

        df = pd.DataFrame(samples)
        df.to_json(path / "formatted.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    # download_data()
    format_data()
