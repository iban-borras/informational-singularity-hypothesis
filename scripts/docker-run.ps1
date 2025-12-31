# ═══════════════════════════════════════════════════════════════════════════
# HSI Agents Project - Docker Run Helper (Windows PowerShell)
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   .\scripts\docker-run.ps1 -v B -i 15           # Generate variant B
#   .\scripts\docker-run.ps1 -v M -i 20 --no-plots # Fibonacci control
#   .\scripts\docker-run.ps1 --help               # Show help
#
# First run will build the image automatically.
# ═══════════════════════════════════════════════════════════════════════════

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"

# Script and project directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# Image name
$ImageName = "hsi-agents:latest"

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host "  HSI Agents Project - Docker Runner" -ForegroundColor Blue
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Blue

# Check if Docker is installed
try {
    docker --version | Out-Null
} catch {
    Write-Host "Error: Docker is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
    exit 1
}

# Check if image exists, build if not
$imageExists = docker image inspect $ImageName 2>$null
if (-not $imageExists) {
    Write-Host "Image not found. Building..." -ForegroundColor Yellow
    Push-Location $ProjectDir
    docker build -t $ImageName .
    Pop-Location
    Write-Host "Image built successfully!" -ForegroundColor Green
}

# Create results directory if it doesn't exist
$ResultsDir = Join-Path $ProjectDir "results"
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

# Run the container
$ArgsString = $Arguments -join " "
Write-Host "Running: python -m hsi_agents_project.level0_generate $ArgsString" -ForegroundColor Green
Write-Host ""

docker run --rm `
    -v "${ProjectDir}\results:/app/hsi_agents_project/results" `
    $ImageName `
    python -m hsi_agents_project.level0_generate @Arguments

Write-Host ""
Write-Host "Done! Results saved to: $ResultsDir" -ForegroundColor Green

