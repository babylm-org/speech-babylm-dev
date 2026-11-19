#!/usr/bin/env bash
# Download the English PhonBank corpora from TalkBank.
set -euo pipefail
uv run python src/speechlm/dataset/download_talkbank_data.py --corpus-name phon --subset-name Eng-NA Eng-UK
