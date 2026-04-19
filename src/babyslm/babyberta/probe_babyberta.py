# Adapted from: https://github.com/phueb/BabyBERTa/blob/master/babyberta/probing.py

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizerFast

from .dataset import DataSet, make_sequences

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name):
    """
    Load tokenizer and model from HuggingFace
    :param model_name:      name of the model that needs to be loaded,
                            must belong to ['BabyBERTa-1', 'BabyBERTa-2', 'BabyBERTa-3']
    :return:                a dictionnary with keys ['tokenizer', 'model']
    """
    assert model_name in ["BabyBERTa-1", "BabyBERTa-2", "BabyBERTa-3"]
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "phueb/%s" % model_name, add_prefix_space=True
    )
    model = RobertaForMaskedLM.from_pretrained("phueb/%s" % model_name).to(DEVICE)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


def prob_extractor_babyberta(model, data):
    """
    Probe BabyBERTa model
    :param model:           a dictionnary with keys ['tokenizer', 'model']
    :param data:            a pandas dataframe with columns ['real', 'fake']
    :return:                cross entropies computed by the BabyBERTa model
    """
    seq_names = data["filename"]
    stimuli = data["transcription"]

    stimuli = make_sequences(stimuli, num_sentences_per_input=1)
    stimuli = ["<s> " + s + " </s>" for s in stimuli]
    stimuli = DataSet.for_probing(stimuli, model["tokenizer"])
    cross_entropies = calc_cross_entropies(model["model"], stimuli)
    cross_entropies = [-c for c in cross_entropies]

    return seq_names, cross_entropies


def calc_cross_entropies(model, dataset):
    """

    :param model:           an instance of RobertaForMaskedLM
    :param dataset:         an instance of DataSet
    :return:                cross entropies computed by the BabyBERTa model
    """
    model.eval()
    cross_entropies = []
    loss_fct = CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for x, _, _ in tqdm(dataset):
            # get loss
            output = model(**{k: v.to(DEVICE) for k, v in x.items()})
            logits_3d = output["logits"]
            logits_for_all_words = logits_3d.permute(0, 2, 1)
            labels = x["input_ids"].to(DEVICE)
            loss = loss_fct(
                logits_for_all_words,  # need to be [batch size, vocab size, seq length]
                labels,  # need to be [batch size, seq length]
            )

            # compute avg cross entropy per sentence
            # to do so, we must exclude loss for padding symbols, using attention_mask
            cross_entropies += [
                loss_i[np.where(row_mask)[0]].mean().item()
                for loss_i, row_mask in zip(loss, x["attention_mask"].numpy())
            ]

    if not cross_entropies:
        raise RuntimeError("Did not compute cross entropies.")

    return cross_entropies
