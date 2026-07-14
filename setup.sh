#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

if ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] not in {(3, 10), (3, 11)})'; then
  echo "Python 3.10 or 3.11 is required for the pinned release environment." >&2
  exit 1
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install \
  torch==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu118

# Prebuilt PyG extension wheels used by the optional large-graph samplers.
"$PYTHON" -m pip install \
  pyg-lib torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.2.1+cu118.html

"$PYTHON" -m pip install -r requirements.txt

echo "Installed the CUDA 11.8 release environment with $("$PYTHON" --version)."
