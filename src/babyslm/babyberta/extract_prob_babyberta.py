import argparse
import json
from pathlib import Path

from .loaders import load_stimuli_text
from .probe_babyberta import load_model, prob_extractor_babyberta


def write_probabilities(seq_names, probabilities, out_file):
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, "w") as f:
        f.writelines(
            f"{filename} {prob}\n" for filename, prob in zip(seq_names, probabilities)
        )
    print(f"Writing pseudo-probabilities to {out_file}")


def write_args(args, out_file):
    out_file.parent.mkdir(exist_ok=True, parents=True)
    args = vars(args)
    args["input_path"] = str(args["input_path"])
    args["model"] = str(args["model"])
    args["out_file"] = str(args["out"])
    with open(out_file, "w") as f:
        json.dump(args, f, indent=2, ensure_ascii=False)
    print(f"Writing args to {out_file}")


def score_babyberta(
    input_path, task, mode="dev", model="babyberta1", out="results/babyslm/babyberta/"
):
    args = argparse.Namespace(
        input_path=Path(input_path),
        mode=mode,
        model=model.replace("babyberta", "BabyBERTa-"),
        task=task,
        out=out,
    )
    type_task = args.input_path.stem
    if args.mode == "both":
        args.mode = ["dev", "test"]
    else:
        args.mode = [args.mode]

    # Load data
    stimuli = load_stimuli_text(args.input_path, args.mode)
    # Load model
    model = load_model(args.model)

    # Compute proba
    for data, data_name in zip(stimuli, args.mode):
        seq_names, probabilities = prob_extractor_babyberta(model, data)
        out_file = Path(args.out) / type_task / f"{data_name}.txt"
        write_probabilities(seq_names, probabilities, out_file)
        args_file = Path(args.out) / type_task / f"args_{data_name}.txt"
        write_args(args, args_file)
