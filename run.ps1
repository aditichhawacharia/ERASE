# PowerShell Setup & Execution Script for ERASE PoC
# Run this script to automatically configure a virtual environment, install requirements, and run the simulation.

# Ensure script halts on error
$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " ERASE Edge Privacy Redaction Simulation Setup " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 1. Verify Python is installed
try {
    $pythonVersion = python --version
    Write-Host "[INFO] Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python was not found on your system. Please install Python 3.8+." -ForegroundColor Red
    Exit 1
}

# 2. Setup Virtual Environment
if (-not (Test-Path -Path ".venv")) {
    Write-Host "[INFO] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "[INFO] Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "[INFO] Existing virtual environment found." -ForegroundColor Green
}

# 3. Activate Virtual Environment & Install Requirements
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
# Activate for the current PowerShell session scope
. .venv\Scripts\Activate.ps1

Write-Host "[INFO] Installing dependencies (this may take a minute or two, downloading PyTorch and OpenCV)..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the Python PoC Script
Write-Host "[INFO] Launching ERASE Edge Camera Simulation..." -ForegroundColor Green
Write-Host "[INFO] Note: The simulation will loop the video. Press 'q' inside the video window to exit." -ForegroundColor Cyan
python erase_poc.py

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Simulation finished." -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
