#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if (( $# > 1 )); then
    printf 'Usage: %s [OUTPUT_DIR]\n' "$0" >&2
    exit 2
fi

OUTPUT_DIR="${1:-${AEGIS_ARTIFACT_RESULTS_DIR:-${PROJECT_ROOT}/artifact-results}}"
RESULT_PATH="${OUTPUT_DIR}/scaled-reproduction.json"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    printf 'ERROR: evaluator environment not found at %s. Run %s first.\n' \
        "${PROJECT_ROOT}/.venv" "${SCRIPT_DIR}/install-artifact.sh" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=''
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=false
export OMP_SCHEDULE=static
export MKL_CBWR=COMPATIBLE
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export COMET_MODE=DISABLED

mkdir -p -- "${OUTPUT_DIR}"
"${VENV_PYTHON}" -m aegis.cli artifact scaled-reproduction \
    --device cpu \
    --output-dir "${OUTPUT_DIR}"
"${VENV_PYTHON}" "${SCRIPT_DIR}/verify-artifact.py" \
    --profile scaled-reproduction \
    "${RESULT_PATH}"
printf 'Scaled-reproduction result: %s\n' "${RESULT_PATH}"
