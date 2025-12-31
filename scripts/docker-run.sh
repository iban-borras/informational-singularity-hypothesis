#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# HSI Agents Project - Docker Run Helper (Linux/Mac)
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/docker-run.sh -v B -i 15           # Generate variant B
#   ./scripts/docker-run.sh -v M -i 20 --no-plots # Fibonacci control
#   ./scripts/docker-run.sh --help               # Show help
#
# First run will build the image automatically.
# ═══════════════════════════════════════════════════════════════════════════

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Image name
IMAGE_NAME="hsi-agents:latest"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  HSI Agents Project - Docker Runner${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if image exists, build if not
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo -e "${YELLOW}Image not found. Building...${NC}"
    cd "$PROJECT_DIR"
    docker build -t "$IMAGE_NAME" .
    echo -e "${GREEN}Image built successfully!${NC}"
fi

# Create results directory if it doesn't exist
mkdir -p "$PROJECT_DIR/results"

# Run the container
echo -e "${GREEN}Running: python -m hsi_agents_project.level0_generate $@${NC}"
echo ""

docker run --rm \
    -v "$PROJECT_DIR/results:/app/hsi_agents_project/results" \
    "$IMAGE_NAME" \
    python -m hsi_agents_project.level0_generate "$@"

echo ""
echo -e "${GREEN}Done! Results saved to: $PROJECT_DIR/results/${NC}"

