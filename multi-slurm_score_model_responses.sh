#!/usr/bin/env bash
# Submit a scoring SLURM job for every JSON file in a directory.
# Usage:
#   ./multi-slurm_score_model_responses.sh <directory> \
#     --partition h100 \
#     --answer_key <answer_key.json> \
#     [--output <output_dir>] \
#     [--reasoning low|medium|high] \
#     [--max-tokens N]
#
# Notes:
#   - For each *.json file directly inside <directory>, submits one job.
#   - An answer key must always be provided via --answer_key. It is passed to
#     all jobs; scoring uses it for filenames ending in "standard.json" or
#     containing "_aad_", "_iasd_", or "_oe_solvable". Other files still score
#     with the unanswerable/none-of-the-above correct answer.
#   - Non-JSON files are ignored. Subdirectories are not traversed.
#   - --partition accepts a30|l40s|h100 or mb-a30|mb-l40s|mb-h100.
#   - --reasoning defaults to low if omitted (same as score_model_responses.py).
#   - --max-tokens defaults to 512 if omitted (same as score_model_responses.py).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/partition_defaults.sh
source "${ROOT}/scripts/partition_defaults.sh"

is_sourced() {
    [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

die() {
    echo "Error: $*" >&2
    return 1
}

usage() {
    echo "Usage: $0 <directory> --partition <a30|l40s|h100|mb-*> --answer_key <answer_key.json> [--output <dir>] [--reasoning low|medium|high] [--max-tokens N]" >&2
}

main() {
    if [ $# -lt 1 ]; then
        usage
        return 1
    fi

    local TARGET_DIR=""
    local ANSWER_KEY=""
    local PARTITION_ARG=""
    local OUTPUT_DIR=""
    local REASONING=""
    local MAX_TOKENS=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --answer_key)
                [ -n "${2:-}" ] || { die "--answer_key requires a path"; return 1; }
                ANSWER_KEY="$2"
                shift 2
                ;;
            --partition)
                [ -n "${2:-}" ] || { die "--partition requires a value"; return 1; }
                PARTITION_ARG="$2"
                shift 2
                ;;
            --output|--output-dir)
                [ -n "${2:-}" ] || { die "--output requires a path"; return 1; }
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --reasoning)
                [ -n "${2:-}" ] || { die "--reasoning requires low|medium|high"; return 1; }
                case "$2" in
                    low|medium|high) ;;
                    *)
                        die "--reasoning must be low, medium, or high (got: $2)"
                        return 1
                        ;;
                esac
                REASONING="$2"
                shift 2
                ;;
            --max-tokens)
                [ -n "${2:-}" ] || { die "--max-tokens requires a positive integer"; return 1; }
                case "$2" in
                    ''|*[!0-9]*)
                        die "--max-tokens must be a positive integer (got: $2)"
                        return 1
                        ;;
                esac
                if [ "$2" -lt 1 ]; then
                    die "--max-tokens must be a positive integer (got: $2)"
                    return 1
                fi
                MAX_TOKENS="$2"
                shift 2
                ;;
            -*)
                die "unknown option: $1"
                return 1
                ;;
            *)
                if [ -n "$TARGET_DIR" ]; then
                    die "unexpected extra argument: $1"
                    return 1
                fi
                TARGET_DIR="$1"
                shift
                ;;
        esac
    done

    [ -n "$TARGET_DIR" ] || { usage; die "<directory> is required"; return 1; }
    [ -n "$ANSWER_KEY" ] || { die "--answer_key <answer_key.json> is required"; return 1; }
    [ -n "$PARTITION_ARG" ] || { die "--partition <a30|l40s|h100|mb-*> is required"; return 1; }

    if [ ! -d "$TARGET_DIR" ]; then
        die "Directory not found: $TARGET_DIR"
        return 1
    fi
    if [ ! -f "$ANSWER_KEY" ]; then
        die "Answer key path does not exist: $ANSWER_KEY"
        return 1
    fi

    resolve_partition_gpus "$PARTITION_ARG" || return 1

    mkdir -p "${ROOT}/slurm_logs"

    echo "Submitting scoring jobs for JSON files in: $TARGET_DIR" >&2
    echo "Answer key: $ANSWER_KEY" >&2
    echo "Partition: ${RESOLVED_PARTITION} (${RESOLVED_GPUS} GPUs, mem=${RESOLVED_MEM})" >&2
    if [ -n "$REASONING" ]; then
        echo "Reasoning: $REASONING" >&2
    fi
    if [ -n "$MAX_TOKENS" ]; then
        echo "Max tokens: $MAX_TOKENS" >&2
    fi
    if [ -n "$OUTPUT_DIR" ]; then
        echo "Output dir: $OUTPUT_DIR" >&2
    fi

    local submitted=0
    local file base

    for file in "$TARGET_DIR"/*.json; do
        if [ ! -e "$file" ]; then
            echo "No JSON files found in $TARGET_DIR" >&2
            break
        fi
        if [ ! -f "$file" ]; then
            continue
        fi

        base="$(basename "$file")"
        echo "Submitting: $base" >&2

        sbatch_cmd=(
            sbatch
            --account=3dllms
            --partition="${RESOLVED_PARTITION}"
            --gres="gpu:${RESOLVED_GPUS}"
            --cpus-per-task="${RESOLVED_CPUS}"
            --mem="${RESOLVED_MEM}"
            --time=7-00:00:00
            --job-name="score_oss120"
            --output="${ROOT}/slurm_logs/score_oss120_%j.out"
            --error="${ROOT}/slurm_logs/score_oss120_%j.out"
            "${ROOT}/slurm_score_model_responses.sh"
            --answer_key "$ANSWER_KEY"
            "$file"
        )
        if [ -n "$OUTPUT_DIR" ]; then
            sbatch_cmd+=(--output "$OUTPUT_DIR")
        fi
        if [ -n "$REASONING" ]; then
            sbatch_cmd+=(--reasoning "$REASONING")
        fi
        if [ -n "$MAX_TOKENS" ]; then
            sbatch_cmd+=(--max-tokens "$MAX_TOKENS")
        fi
        "${sbatch_cmd[@]}"
        submitted=$((submitted + 1))
        sleep 0.2
    done

    echo "Done. Submitted: $submitted" >&2
}

if is_sourced; then
    main "$@" || return $?
else
    main "$@" || exit $?
fi
