# syntax=docker/dockerfile:1.7
#
# Reproducible CPU image for ECG-Biometrics-Bench.
#
# Build:
#   docker build -t ecg-biometrics-bench .
#
# Run the test suite:
#   docker run --rm ecg-biometrics-bench python -m pytest tests -q
#
# Run an experiment, mounting host directories so datasets are downloaded
# once and results survive the container:
#   docker run --rm \
#       -v "$(pwd)/datasets:/app/datasets" \
#       -v "$(pwd)/artifacts:/artifacts" \
#       ecg-biometrics-bench \
#       python main.py --config configs/paper_reproduction/ecgid/ecgid_all_available_closed_set_task01_identification.yaml
#
# This image installs the CPU build of PyTorch, which keeps it near 1 GB
# instead of several. For GPU execution, run on the host with the CUDA build
# described in README.md, or derive an image from an nvidia/cuda base.

# ---------------------------------------------------------------- builder ---
# Dependencies are compiled and installed in a throwaway stage so that no
# build toolchain reaches the runtime image.
FROM python:3.10-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed by source-only wheels; it is discarded with this
# stage rather than shipped.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Requirements are copied before the source so that the dependency layer is
# reused whenever only application code changes. This is the difference
# between a five-second rebuild and a five-minute one.
COPY requirements.txt requirements-dev.txt ./

# The CPU wheel index keeps the image small; the pins are build-agnostic and
# resolve against it unchanged.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt -r requirements-dev.txt

# ---------------------------------------------------------------- runtime ---
FROM python:3.10-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="ECG-Biometrics-Bench" \
      org.opencontainers.image.description="Reproducible benchmarking framework for ECG biometric recognition" \
      org.opencontainers.image.source="https://github.com/MParvan/ecg-biometrics-bench" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg

# libgomp is required by the PyTorch and scikit-learn runtimes; the rest of
# the build toolchain is deliberately absent.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user. A container that writes results should not be
# writing them as root.
RUN useradd --create-home --uid 1000 benchmark

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=benchmark:benchmark . .

# Datasets are downloaded on first use and results are written outside the
# source tree, so both are mount points rather than image content.
#
# /app itself is chowned as well as its contents: WORKDIR creates it as root,
# and a directory the runtime user cannot write is enough to break tools that
# expect to create scratch files beside the source.
RUN mkdir -p /app/datasets /artifacts \
    && chown benchmark:benchmark /app \
    && chown -R benchmark:benchmark /app/datasets /artifacts

USER benchmark

# Fail the build if the framework cannot be imported, so a broken image is
# never published.
RUN python -c "import main, run, load_dataset, models, utils; print('framework imports OK')"

ENTRYPOINT ["python"]
CMD ["main.py", "--help"]
