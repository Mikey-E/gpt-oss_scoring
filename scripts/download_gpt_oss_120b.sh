#!/usr/bin/env bash
# Download the Hugging Face MXFP4 GPT-OSS-120B checkpoint into this repo.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="openai/gpt-oss-120b"
DEST="${ROOT}/models/gpt-oss-120b"

CONDA_SH="${CONDA_INSTALL_PATH:-/project/3dllms/melgin/conda}/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    if conda env list | grep -qE '^gpt-oss-scoring\s'; then
        conda activate gpt-oss-scoring
    fi
fi

mkdir -p "${DEST}"
echo "Downloading ${MODEL_ID} to ${DEST}"

if command -v hf >/dev/null 2>&1; then
    hf download "${MODEL_ID}" \
        --exclude "metal/*" "original/*" \
        --local-dir "${DEST}"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${MODEL_ID}" \
        --exclude "metal/*" "original/*" \
        --local-dir "${DEST}"
else
    echo "Hugging Face CLI not found. Install huggingface_hub first." >&2
    exit 1
fi

python - "${DEST}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
index_path = model_dir / "model.safetensors.index.json"
if not (model_dir / "config.json").is_file() or not index_path.is_file():
    raise SystemExit("Download incomplete: config or weight index is missing")

index = json.loads(index_path.read_text())
missing = sorted(
    shard for shard in set(index["weight_map"].values()) if not (model_dir / shard).is_file()
)
if missing:
    raise SystemExit(f"Download incomplete: {len(missing)} weight shards are missing")

print(f"Verified {len(set(index['weight_map'].values()))} weight shards.")
PY

du -sh "${DEST}"
echo "Ready."
