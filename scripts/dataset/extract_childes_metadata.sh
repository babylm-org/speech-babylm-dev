#!/usr/bin/env bash
# Extract the CHILDES metadata (speakers, roles, ages) to results/dataset/.
set -euo pipefail
uv run python src/speechlm/dataset/extract_metadata.py --output_path results/dataset/childes_metadata.csv
