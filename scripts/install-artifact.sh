#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
VENV_DIR="${PROJECT_ROOT}/.venv"
CONSTRAINTS="${PROJECT_ROOT}/constraints-cpu.txt"
PYTHON_BIN="${AEGIS_PYTHON:-python3.11}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf 'ERROR: Python 3.11 was not found as %q. Set AEGIS_PYTHON to a Python 3.11 executable.\n' "${PYTHON_BIN}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
    printf 'ERROR: %q is not Python 3.11. Set AEGIS_PYTHON to a Python 3.11 executable.\n' "${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${CONSTRAINTS}" ]]; then
    printf 'ERROR: dependency constraints not found: %s\n' "${CONSTRAINTS}" >&2
    exit 1
fi

if [[ -e "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" ]]; then
    printf 'ERROR: %s exists but is not a usable virtual environment. Move or remove it first.\n' "${VENV_DIR}" >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    printf 'Creating Python 3.11 virtual environment at %s\n' "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
if ! "${VENV_PYTHON}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
    printf 'ERROR: existing virtual environment is not Python 3.11: %s\n' "${VENV_DIR}" >&2
    exit 1
fi

printf 'Installing pinned evaluator dependencies\n'
"${VENV_PYTHON}" -m pip install --upgrade \
    'pip==24.3.1' \
    'setuptools==75.6.0' \
    'wheel==0.45.1'

case "$(uname -s)" in
    Linux)
        "${VENV_PYTHON}" -m pip install \
            --index-url 'https://download.pytorch.org/whl/cpu' \
            --extra-index-url 'https://pypi.org/simple' \
            --constraint "${CONSTRAINTS}" \
            torch
        ;;
    Darwin)
        "${VENV_PYTHON}" -m pip install \
            --constraint "${CONSTRAINTS}" \
            torch
        ;;
    *)
        printf 'ERROR: supported evaluator platforms are Linux and macOS; detected %s.\n' "$(uname -s)" >&2
        exit 1
        ;;
esac

"${VENV_PYTHON}" -m pip install \
    --constraint "${CONSTRAINTS}" \
    accelerate \
    datasets \
    huggingface-hub \
    nltk \
    numpy \
    peft \
    pillow \
    rouge-score \
    safetensors \
    scikit-learn \
    scipy \
    tokenizers \
    tqdm \
    transformers

printf 'Installing AEGIS from %s\n' "${PROJECT_ROOT}"
"${VENV_PYTHON}" -m pip install --no-build-isolation --no-deps "${PROJECT_ROOT}"
"${VENV_PYTHON}" -m pip check

"${VENV_PYTHON}" - <<'PY'
import datasets
import numpy
import torch
import transformers

expected = {
    "torch": "2.5.1",
    "transformers": "4.46.3",
    "datasets": "2.21.0",
    "numpy": "1.26.4",
}
actual = {
    "torch": torch.__version__.split("+", 1)[0],
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "numpy": numpy.__version__,
}
if actual != expected:
    raise SystemExit(f"ERROR: installed evaluator versions differ: expected={expected}, actual={actual}")
if torch.version.cuda is not None:
    raise SystemExit(f"ERROR: CUDA-enabled torch was installed ({torch.version.cuda}); CPU-only torch is required")
print("Evaluator environment ready:", ", ".join(f"{name}={version}" for name, version in actual.items()))
PY

printf 'Installation complete. Run %s\n' "${PROJECT_ROOT}/scripts/kick-the-tires.sh"
