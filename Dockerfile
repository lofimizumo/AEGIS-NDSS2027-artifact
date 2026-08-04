FROM python:3.11-slim

ENV VIRTUAL_ENV=/opt/aegis/.venv \
    PATH=/opt/aegis/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    CUDA_VISIBLE_DEVICES="" \
    NVIDIA_VISIBLE_DEVICES=void \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_DYNAMIC=false \
    OMP_SCHEDULE=static \
    MKL_CBWR=COMPATIBLE \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    WANDB_DISABLED=true \
    WANDB_MODE=disabled \
    COMET_MODE=DISABLED \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_INPUT=1

WORKDIR /opt/aegis

COPY constraints-cpu.txt pyproject.toml THIRD_PARTY_NOTICES.md ./
COPY LICENSES/ ./LICENSES/
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN chmod 0755 scripts/*.sh scripts/verify-artifact.py \
    && AEGIS_PYTHON=python3.11 scripts/install-artifact.sh

RUN groupadd --system aegis \
    && useradd --system --gid aegis --create-home --home-dir /home/aegis aegis \
    && mkdir -p /opt/aegis/artifact-results /home/aegis/.cache \
    && chown -R aegis:aegis /opt/aegis/artifact-results /home/aegis

ENV HOME=/home/aegis \
    XDG_CACHE_HOME=/home/aegis/.cache \
    HF_HOME=/home/aegis/.cache/huggingface

USER aegis

CMD ["/opt/aegis/scripts/kick-the-tires.sh"]
