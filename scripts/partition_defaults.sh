#!/usr/bin/env bash
# Resolve short GPU aliases or full Slurm partition names to partition + GPU count.
# Usage: source this file, then: resolve_partition_gpus "$PARTITION_ARG"
# Sets: RESOLVED_PARTITION, RESOLVED_GPUS, RESOLVED_MEM, RESOLVED_CPUS

resolve_partition_gpus() {
    local raw="${1:-}"
    local key
    key="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"

    case "$key" in
        a30|mb-a30)
            RESOLVED_PARTITION="mb-a30"
            RESOLVED_GPUS=4
            RESOLVED_MEM="256G"
            RESOLVED_CPUS=32
            ;;
        l40s|mb-l40s)
            RESOLVED_PARTITION="mb-l40s"
            RESOLVED_GPUS=2
            RESOLVED_MEM="192G"
            RESOLVED_CPUS=16
            ;;
        h100|mb-h100)
            RESOLVED_PARTITION="mb-h100"
            RESOLVED_GPUS=1
            RESOLVED_MEM="128G"
            RESOLVED_CPUS=8
            ;;
        *)
            echo "Error: unsupported partition '$raw'." >&2
            echo "Use one of: a30, l40s, h100, mb-a30, mb-l40s, mb-h100" >&2
            return 1
            ;;
    esac
    return 0
}
