#!/usr/bin/env bash
# Download the English CHILDES corpora from TalkBank.
set -euo pipefail
uv run python src/speechlm/dataset/download_talkbank_data.py --corpus-name childes --subset-name Eng-AAE Eng-NA Eng-UK
