#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-${repository_root}/_site}"

if [[ "$output_dir" == "/" || "$output_dir" == "$repository_root" ]]; then
  echo "Refusing to use an unsafe website output directory: $output_dir" >&2
  exit 2
fi

mkdir -p "$output_dir/assets"
cp "$repository_root/website/index.html" "$output_dir/index.html"
cp "$repository_root/website/styles.css" "$output_dir/styles.css"
cp "$repository_root/website/script.js" "$output_dir/script.js"
cp "$repository_root/assets/intro_new.png" "$output_dir/assets/intro_new.png"
cp "$repository_root/assets/method.png" "$output_dir/assets/method.png"
touch "$output_dir/.nojekyll"
