import pandas as pd


def load_stimuli_text(path, kinds):
    out = []
    if path.stem.startswith("lexical"):
        key = "phones"
    else:
        key = "transcription"

    for kind in kinds:
        stimuli_path = path / kind / "gold.csv"
        data = pd.read_csv(stimuli_path)
        voices = data["voice"].unique()
        data = data[data["voice"] == voices[0]][[key, "filename"]]
        data.columns = ["transcription", "filename"]
        out.append(data)
    print("Stimuli loaded.")
    return out
