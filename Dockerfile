# Reproducibility-capsule image for preorder4mlc.
# Reproduces the CHD-49 result on CPU.
FROM python:3.10-slim

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran \
    && rm -rf /var/lib/apt/lists/*

# Pin the core scientific stack first so the package install does not re-resolve.
COPY requirements-core.txt /app/requirements-core.txt
RUN pip install -r /app/requirements-core.txt

# Install the package itself without disturbing the pinned dependencies.
COPY . /app
RUN pip install --no-deps -e .

# Default reproducible run: CHD-49 -> /results (mount /results to retrieve output).
CMD ["bash", "scripts/reproduce_capsule.sh", "/results"]
