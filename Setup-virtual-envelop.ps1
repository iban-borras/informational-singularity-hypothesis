#
# Setup script for HSI Agents Project environment
# Hipòtesi de Singularitat Informacional
#

# Configure PowerShell to use UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
try { chcp 65001 | Out-Null } catch { Write-Host "Unable to change code page to UTF-8" }

# Enable execution policy for this script
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Configure UTF-8 encoding for Windows (solves Unicode issues)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8
} catch {
    # Ignore if console encoding can't be set
}

# Configure variables
$pythonVersion = "3.11.8"
$virtualEnvPath = ".\venv"  # Canviat a una ruta més curta per evitar problemes de permisos
$logFile = ".\setup_log.txt"
$requirementsFile = ".\requirements.txt"  # Ruta corregida per HSI

# Function to write to log and console
function Write-Log {
    param (
        [string]$Message,
        [switch]$IsError
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"

    if ($IsError) {
        Write-Host $logMessage -ForegroundColor Red
        Add-Content -Path $logFile -Value "ERROR: $logMessage"
    } else {
        Write-Host $logMessage -ForegroundColor Green
        Add-Content -Path $logFile -Value "$logMessage"
    }
}

# Start installation
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  HSI Agents Project Setup " -ForegroundColor Cyan
Write-Host "  Hipòtesi de Singularitat Informacional " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Create log
Write-Log "Starting installation..."

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Log "Python is already installed: $pythonVersion"
} catch {
    Write-Log "Python is not installed. Installing Python $pythonVersion..." -IsError
    Write-Log "Please install Python manually from python.org" -IsError
    Exit 1
}

# Verificar si el requirements.txt existeix
if (-not (Test-Path -Path $requirementsFile)) {
    Write-Log "Requirements file not found at $requirementsFile" -IsError
    Write-Log "Please make sure the file exists and the path is correct" -IsError
    Exit 1
}

# Verificar si ya existe un entorno virtual funcional
$needsNewVenv = $false
if (Test-Path -Path $virtualEnvPath) {
    Write-Log "Existing virtual environment detected. Checking if it's functional..."

    # Verificar si el script de activación existe
    if (Test-Path -Path "$virtualEnvPath\Scripts\Activate.ps1") {
        try {
            # Intentar activar el entorno existente
            & ".\$virtualEnvPath\Scripts\Activate.ps1"

            # Verificar si Python funciona correctamente
            $pythonPath = python -c "import sys; print(sys.executable)" 2>$null
            if ($pythonPath -like "*venv*") {
                Write-Log "Existing virtual environment is functional. Reusing it."
                $needsNewVenv = $false
            } else {
                Write-Log "Existing virtual environment is not working properly."
                $needsNewVenv = $true
            }
        } catch {
            Write-Log "Error testing existing virtual environment: $_"
            $needsNewVenv = $true
        }
    } else {
        Write-Log "Activation script not found in existing virtual environment."
        $needsNewVenv = $true
    }

    if ($needsNewVenv) {
        Write-Log "Removing corrupted virtual environment..."
        Remove-Item -Path $virtualEnvPath -Recurse -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
} else {
    Write-Log "No existing virtual environment found."
    $needsNewVenv = $true
}

# Create virtual environment only if needed
if ($needsNewVenv) {
    Write-Log "Creating new virtual environment in $virtualEnvPath..."
    try {
        # Use venv (built into Python)
        python -m venv $virtualEnvPath --clear

        # Verify if created successfully
        if (Test-Path -Path "$virtualEnvPath\Scripts\Activate.ps1") {
            Write-Log "Virtual environment created successfully."
        } else {
            throw "Activation script not found."
        }
    } catch {
        Write-Log "Error creating virtual environment: $_" -IsError
        Exit 1
    }
}

# Activate virtual environment (if not already activated)
if (-not $env:VIRTUAL_ENV -or $needsNewVenv) {
    Write-Log "Activating virtual environment..."
    try {
        # Activate virtual environment with relative path
        & ".\$virtualEnvPath\Scripts\Activate.ps1"

        if ($env:VIRTUAL_ENV) {
            Write-Log "Virtual environment activated successfully."
        } else {
            throw "Failed to activate virtual environment"
        }
    } catch {
        Write-Log "Error activating virtual environment: $_" -IsError
        Exit 1
    }
} else {
    Write-Log "Virtual environment already activated."
}

# Install dependencies
Write-Log "Installing dependencies..."
try {
    # Update pip
    python -m pip install --upgrade pip

    # Check if requirements file exists
    if (Test-Path -Path $requirementsFile) {
        # Install dependencies from requirements.txt file
        python -m pip install -r $requirementsFile
        Write-Log "Dependencies installed successfully."
    } else {
        throw "Requirements file not found at $requirementsFile"
    }
} catch {
    Write-Log "Error installing dependencies: $_" -IsError
    Exit 1
}
# Ensure .env exists (copy from template if missing) and preview HSI_* keys
try {
    $envPath = ".\.env"
    $templateCandidates = @(".\.env.template", ".\hsi_agents_project\.env.template")
    if (-not (Test-Path -Path $envPath)) {
        foreach ($tpl in $templateCandidates) {
            if (Test-Path -Path $tpl) {
                Copy-Item -Path $tpl -Destination $envPath -Force
                Write-Log "Created .env from template: $tpl"
                break
            }
        }
    }

    if (Test-Path -Path $envPath) {
        Write-Host "Preview of HSI_* keys from .env (adjust as needed):" -ForegroundColor Yellow
        Get-Content $envPath | ForEach-Object {
            $line = $_.Trim()
            if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
                $pair = $line.Split("=",2)
                $k = $pair[0].Trim()
                $v = $pair[1].Trim()
                if ($k.StartsWith("HSI_")) { Write-Host ("  {0}={1}" -f $k,$v) -ForegroundColor White }
            }
        }
    } else {
        Write-Log ".env file not found; you can create it from .env.template"
    }
} catch {
    Write-Log "Error handling .env/.env.template: $_" -IsError
}



# Finish
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "    HSI Setup completed successfully!       " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the HSI experiment:" -ForegroundColor Yellow
Write-Host "  python main.py                    # Run full experiment" -ForegroundColor White
Write-Host "  python main.py --help             # Show all options" -ForegroundColor White
Write-Host "  python level0/generator.py        # Test Phi generator" -ForegroundColor White
Write-Host "  python run_all_variants.py        # Test all HSI variants" -ForegroundColor White
Write-Host ""
Write-Host "Results will be saved in: ./results/" -ForegroundColor Cyan
Write-Host ""
Write-Host "UTF-8 encoding configured for Unicode support!" -ForegroundColor Magenta
Write-Host "Ready to explore the Informational Singularity!" -ForegroundColor Green
Write-Host ""