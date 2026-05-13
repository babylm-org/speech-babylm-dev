import os
from pathlib import Path


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


if __name__ == "__main__":
    # download_data()
    format_data()
