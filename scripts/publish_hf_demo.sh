#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
space_id="${1:-AIGNLAI/RouteFM-Demo}"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required. Install huggingface_hub and run 'hf auth login'." >&2
  exit 2
fi

hf repo create "$space_id" --repo-type space --space-sdk gradio --exist-ok
hf upload "$space_id" "$repository_root/demo" . --repo-type space \
  --exclude "__pycache__/*" --exclude "*.pyc"
