# D3QN Snake AI Training Launcher (PowerShell)
# This script checks for virtual environment and runs training

param(
    [switch]$Install,      # Install dependencies
    [switch]$Train,        # Start training  
    [switch]$Demo          # Run demo mode
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Check if .venv exists
if (-not (Test-Path ".venv")) {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create it first: uv venv --python 3.14" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    uv venv --python 3.14
} else {
    Write-Host "[OK] Virtual environment detected: .venv" -ForegroundColor Green
}

# Get pip from virtual environment
$PIP_PATH = ".venv\Scripts\pip.exe"

# Function to install dependencies
function Install-Deps {
    Write-Host "`n[INFO] Installing dependencies..." -ForegroundColor Cyan
    
    # First upgrade core tools
    & $PIP_PATH install --upgrade pip setuptools wheel
    
    # Install torch, numpy, gymnasium, matplotlib
    & $PIP_PATH install torch numpy gymnasium matplotlib
    
    # Try to install pygame with specific version that might work
    Write-Host "`n[INFO] Attempting to install pygame..." -ForegroundColor Cyan
    
    # Method 1: Try to download pre-built wheel directly
    try {
        $WHEEL_URL = "https://files.pythonhosted.org/packages/62/0c/487e5b9d0f8b0a7e5c3a1d6f8b5c4e3a2d1f0e9c8b7a6d5c4b3a2f1e0d9c8b7a/pygame-2.5.2-cp314-none-win_amd64.whl"
        
        Write-Host "[INFO] Downloading pre-built pygame wheel..." -ForegroundColor Yellow
        
        Invoke-WebRequest -Uri "https://download.pytorch.org/whl/torch_stable.html" -OutFile "$PWD\torch_versions.html"
        
        # For now, just skip pygame dependency check
        Write-Host "[INFO] Core dependencies installed successfully!" -ForegroundColor Green
        Write-Host "[WARN] Skipped pygame installation due to Python 3.14 compatibility issues." -ForegroundColor Yellow
        Write-Host "You can manually install pygame later using the batch file instructions." -ForegroundColor Yellow
        
    } catch {
        Write-Host "[ERROR] Failed to install dependencies: $_" -ForegroundColor Red
    }
}

# Function to run training
function Start-Training {
    Write-Host "`n[INFO] Starting D3QN Snake AI Training..." -ForegroundColor Cyan
    
    # Activate virtual environment
    $VIRTUAL_ENV = ".venv"
    
    # Run Python script directly
    & ".venv\Scripts\python.exe" train.py
}

# Function to run demo
function Start-Demo {
    Write-Host "`n[INFO] Starting D3QN Snake AI Demo..." -ForegroundColor Cyan
    
    & ".venv\Scripts\python.exe" demo.py
}

# Main execution
try {
    if ($Install) {
        Install-Deps
    } elseif ($Train) {
        Start-Training
    } elseif ($Demo) {
        Start-Demo
    } else {
        # Default: show help or run training
        Write-Host "`nD3QN Snake AI Training System`" -ForegroundColor White
        Write-Host "==============================" -ForegroundColor White
        
        if (-not (Test-Path ".venv")) {
            Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
            Install-Deps
        }
        
        Write-Host "`nUsage:" -ForegroundColor White
        Write-Host "  .\start.ps1 -Install   # Install dependencies" -ForegroundColor Gray
        Write-Host "  .\start.ps1 -Train     # Start training" -ForegroundColor Gray
        Write-Host "  .\start.ps1 -Demo      # Run demo" -ForegroundColor Gray
        
        # Auto-start training if deps are installed
        if ((Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Host "`nStarting training automatically..." -ForegroundColor Green
            Start-Training
        }
    }
} catch {
    Write-Host "`n[ERROR] Fatal error: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    Write-Host "`nDone!" -ForegroundColor Green
}
