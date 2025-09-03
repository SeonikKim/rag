#!/usr/bin/env bash
set -euo pipefail
PDF="$1"
OUT="${2:-./out}"
CFG="${3:-./configs/config.yaml}"

python ingest.py --pdf "$PDF" --out "$OUT" --config "$CFG"
