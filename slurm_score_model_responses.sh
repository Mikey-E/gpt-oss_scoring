#!/usr/bin/env bash
# Submit (or run inside) a gpt-oss-120b scoring job for one unscored JSON file.
#
# Preferred usage from a login node:
#   ./slurm_score_model_responses.sh --partition h100 \
#     --answer_key /path/to/answer_key.json /path/to/unscored.json
#   ./slurm_score_model_responses.sh --partition mb-l40s \
#     --answer_key /path/to/answer_key.json --output /path/to/outdir /path/to/unscored.json
#
# Or via sbatch with resources on the sbatch command line:
#   sbatch --partition=mb-h100 --gres=gpu:1 --mem=128G --cpus-per-task=8 \
#     slurm_score_model_responses.sh --answer_key KEY.json FILE.json

#SBATCH --account=3dllms
#SBATCH --job-name=score_oss120
#SBATCH --output=./slurm_logs/score_oss120_%j.out
#SBATCH --error=./slurm_logs/score_oss120_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --export=NIL

set -euo pipefail

# Slurm copies the batch script into /var/spool/slurmd/job*/; BASH_SOURCE then
# points there, so fall back to SLURM_SUBMIT_DIR (cwd at sbatch time).
resolve_repo_root() {
    local candidate
    candidate="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${candidate}/score_model_responses.py" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/score_model_responses.py" ]]; then
        printf '%s\n' "$SLURM_SUBMIT_DIR"
        return 0
    fi
    echo "Error: cannot locate gpt-oss_scoring repo root (tried ${candidate} and SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})" >&2
    return 1
}

ROOT="$(resolve_repo_root)" || exit 1
# shellcheck source=scripts/partition_defaults.sh
source "${ROOT}/scripts/partition_defaults.sh"

PARTITION_ARG=""
ANSWER_KEY=""
OUTPUT_DIR=""
JSON_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)
            [[ -n "${2:-}" ]] || { echo "Error: --partition requires a value" >&2; exit 1; }
            PARTITION_ARG="$2"
            shift 2
            ;;
        --answer_key)
            [[ -n "${2:-}" ]] || { echo "Error: --answer_key requires a path" >&2; exit 1; }
            ANSWER_KEY="$2"
            shift 2
            ;;
        --output|--output-dir)
            [[ -n "${2:-}" ]] || { echo "Error: --output requires a path" >&2; exit 1; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -*)
            EXTRA_ARGS+=("$1")
            if [[ $# -gt 1 && "${2:-}" != -* ]]; then
                EXTRA_ARGS+=("$2")
                shift 2
            else
                shift
            fi
            ;;
        *)
            if [[ -n "$JSON_FILE" ]]; then
                echo "Error: unexpected extra argument: $1" >&2
                exit 1
            fi
            JSON_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "$ANSWER_KEY" ]]; then
    echo "Error: --answer_key <path> is required." >&2
    exit 1
fi
if [[ -z "$JSON_FILE" ]]; then
    echo "Error: path to unscored JSON file is required." >&2
    exit 1
fi

# Outside Slurm: require --partition and self-submit with the right resources.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -z "$PARTITION_ARG" ]]; then
        echo "Error: --partition {a30,l40s,h100|mb-a30,mb-l40s,mb-h100} is required." >&2
        exit 1
    fi
    resolve_partition_gpus "$PARTITION_ARG" || exit 1
    mkdir -p "${ROOT}/slurm_logs"
    echo "Submitting scoring job for $(basename "$JSON_FILE") on ${RESOLVED_PARTITION} (${RESOLVED_GPUS} GPUs)" >&2
    sbatch_args=(
        --account=3dllms
        --partition="${RESOLVED_PARTITION}"
        --gres="gpu:${RESOLVED_GPUS}"
        --cpus-per-task="${RESOLVED_CPUS}"
        --mem="${RESOLVED_MEM}"
        --time=7-00:00:00
        --job-name=score_oss120
        --output="${ROOT}/slurm_logs/score_oss120_%j.out"
        --error="${ROOT}/slurm_logs/score_oss120_%j.out"
    )
    cmd=(
        "$0"
        --answer_key "$ANSWER_KEY"
        "$JSON_FILE"
    )
    if [[ -n "$OUTPUT_DIR" ]]; then
        cmd+=(--output "$OUTPUT_DIR")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi
    exec sbatch "${sbatch_args[@]}" "${cmd[@]}"
fi

# Inside allocation.
if [[ -n "$PARTITION_ARG" ]]; then
    resolve_partition_gpus "$PARTITION_ARG" || true
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "ERROR: No GPU visible in this job. Submit via ./slurm_score_model_responses.sh --partition ..." >&2
    echo "or pass --gres/--partition on the sbatch command line." >&2
    exit 1
fi

CONDA_SH="${CONDA_INSTALL_PATH:-/project/3dllms/melgin/conda}/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: conda.sh not found at $CONDA_SH" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate gpt-oss-scoring

cd "$ROOT"
mkdir -p slurm_logs

echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true

CMD=(
    python score_model_responses.py
    "$JSON_FILE"
    --answer_key "$ANSWER_KEY"
)
if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output "$OUTPUT_DIR")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
