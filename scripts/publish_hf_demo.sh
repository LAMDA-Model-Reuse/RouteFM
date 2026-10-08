#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-static}"
space_id="${2:-AIGNLAI/RouteFM-Demo}"

case "$mode" in
  static) source_dir="$repository_root/static_demo"; sdk="static" ;;
  gradio) source_dir="$repository_root/demo"; sdk="gradio" ;;
  *) echo "Usage: $0 [static|gradio] [namespace/name]" >&2; exit 2 ;;
esac

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required. Install huggingface_hub and run 'hf auth login'." >&2
  exit 2
fi

hf repo create "$space_id" --repo-type space --space-sdk "$sdk" --exist-ok
hf upload "$space_id" "$source_dir" . --repo-type space \
  --exclude "__pycache__/*" --exclude "*.pyc"
