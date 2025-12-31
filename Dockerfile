# ═══════════════════════════════════════════════════════════════════════════
# HSI Agents Project - Docker Image
# Informational Singularity Hypothesis (ISH) Research Platform
# ═══════════════════════════════════════════════════════════════════════════
#
# Build:  docker build -t hsi-agents .
# Run:    docker run -v ./results:/app/hsi_agents_project/results hsi-agents \
#             python -m hsi_agents_project.level0_generate -v B -i 15
#
# ═══════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies for scientific computing
# - gcc/g++: Required for compiling numpy/scipy wheels on some platforms
# - gzip: For compressed phi snapshots
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . /app/hsi_agents_project/

# Create results directory structure
RUN mkdir -p /app/hsi_agents_project/results/level0/phi_snapshots \
    && mkdir -p /app/hsi_agents_project/results/level0/reports \
    && mkdir -p /app/hsi_agents_project/results/level0/visualizations \
    && mkdir -p /app/hsi_agents_project/results/level1 \
    && mkdir -p /app/hsi_agents_project/results/level2 \
    && mkdir -p /app/hsi_agents_project/results/cache \
    && mkdir -p /app/hsi_agents_project/results/temp

# Set Python path
ENV PYTHONPATH=/app

# Default command: show help
CMD ["python", "-m", "hsi_agents_project.level0_generate", "--help"]

