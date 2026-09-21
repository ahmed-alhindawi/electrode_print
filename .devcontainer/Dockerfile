FROM docker.io/ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# 1. System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential pkg-config tk-dev openssh-client \
    && rm -rf /var/lib/apt/lists/*

# 2. uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Environment
ENV UV_NO_PROGRESS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/workspace/src \
    OMT_DATA=/datasets/sharm \
    MPLBACKEND=Agg

WORKDIR /tmp/build
COPY pyproject.toml ./
# uv.lock is optional: the bracket glob makes COPY a no-op when it is absent
# COPY uv.loc[k] ./

ARG PYTHON_VERSION=3.13

# 4. Dependencies. Locks if no lock file is present, then syncs frozen.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install ${PYTHON_VERSION} && \
    uv venv /opt/venv --python ${PYTHON_VERSION} && \
    ([ -f uv.lock ] || uv lock) && \
    uv sync --frozen --no-install-project

WORKDIR /workspace
