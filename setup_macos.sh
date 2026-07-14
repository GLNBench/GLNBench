#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

if ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] not in {(3, 10), (3, 11)})'; then
  echo "Python 3.10 or 3.11 is required for the pinned release environment." >&2
  exit 1
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

# requirements.txt installs the matching CPU torch-scatter and torch-sparse wheels used by GCOD's
# NeighborLoader path. Other compiled PyG extensions are optional and can be
# installed from the same PyG wheel matrix for additional large-graph samplers.

echo "Installed the macOS release environment with $("$PYTHON" --version)."
