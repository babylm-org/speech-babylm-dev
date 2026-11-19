#!/usr/bin/env bash
# Extract the CHILDES metadata (speakers, roles, ages) to src/speechlm/dataset/output/.
set -euo pipefail
uv run python src/speechlm/dataset/extract_metadata.py --output_path src/speechlm/dataset/output/childes_metadata.csv
