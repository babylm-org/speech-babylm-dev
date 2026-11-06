# General instructions

## TL;DR
The goal of the _Speech_ BabyLM challenge is to build a capable spoken language model from developmentally realistic speech data.

Participating in the challenge in the challenge involves two tasks:
  
1. Training a speech _encoder_ (speech2unit model): this model maps from raw audio waveforms to units used for language modelling.
2. Training a speech _language model_ (unit-lm): this is a sequence model which can output probabilities over speech-unit sequences.

Submissions to the challenge can either participate in both tasks (1+2), or make use of our pre-trained encoder baselines and focus on the language modelling task (2).
We provide a dataset of speech realistically spoken around children, extracted from the CHILDES corpus. This dataset amounts to XXXX hours of audio and includes XXXX words.
For training models, participants are allowed to supplement this data with non-speech audio, but the total word count must remain the same.

Below we provide more information about the two tasks and evaluation pipeline.

## Task 1: Speech encoding
How should the speech signal be represented? 

Encoder models map from raw waveforms to high-dimensional representations, and can be used for (discrete) unit extraction. This provides the input representations that speech LMs operate over.
Examples of widely used speech encoder models are [Wav2Vec2](https://arxiv.org/abs/2006.11477), [HuBERT](https://arxiv.org/abs/2106.07447), and [SpidR](https://arxiv.org/abs/2512.20308).

A speech encoder submission to the challenge should take in an audio **waveform** and output a **sequence of units** at a sub-phonemic timescale (XXXX ms per unit).

This could for example exist of the following:

**1.a)** A trained Wav2Vec2 model taking in audio waveforms and generating 768-dimensional frame embeddings at a frame rate of 50 Hz (i.e. one representation for every 20 ms of audio).

**1.b)** A trained kmeans model mapping Wav2Vec2's representations to discrete kmeans labels.

For participants who wish to only participate in task 2, we provide pre-trained HuBERT and kmeans models here:
* XXXX link to pre-trained HuBERT checkpoint
* XXXX link to pre-trained kmeans model

It is also possible to participate in task 2 only while using an alternative method for unit extraction (replacing kmeans). 
In this case, download only the HuBERT speech encoder and include your unit extraction module as part of your submission.

## Task 2: Speech language modelling
How do we learn generative speech models?

Speech language models take in a sequence of units representing the speech signal and can be used 1) to estimate probabilities for particular sequences, 2) to generate upcoming units.

Existing speech language models for example include models from the [SLAM suite](https://pages.cs.huji.ac.il/adiyoss-lab/slamming/), and [GSLM2](https://arxiv.org/abs/2611.01143).

A speech language model submission to the challenge should take in unit sequences from our pre-trained encoder baselines OR your own encoder model, and output sequence probabilities (or log-likelihoods).

## Evaluations

We include a set of common evaluations both for assessing speech encoder representations (task 1) and for speech language modelling (task 2).

These include
* for speech encoder reprsentations: phoneme ABX, prosodic ABX, syntactic/semantic ABX, ...
* for speech langauge modelling: sWUGGY, sBLIMP, tSC, ProsAudit, ...

Find more instructions for the evaluation pipeline here: XXXX
