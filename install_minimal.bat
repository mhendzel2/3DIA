@echo off
REM PyMaris - Minimal Installation (No C++ Build Tools Required)
REM This script installs only packages that don't require compilation

echo ========================================
echo PyMaris - Minimal Installation
echo (No C++ Build Tools Required)
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Removing old venv...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install core dependencies (pure Python or pre-compiled wheels)
echo [5/5] Installing core dependencies...
echo This installation avoids packages that require C++ compilation
echo.

REM Install napari with PyQt (pre-compiled wheels available)
echo Installing napari and GUI...
pip install napari[pyqt6]
if %errorlevel% neq 0 (
    echo Trying PyQt5 instead...
    pip install napari[pyqt5]
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install napari
        pause
        exit /b 1
    )
)

REM Install scientific packages (all have pre-compiled wheels on Windows)
echo Installing scientific computing packages...
pip install numpy scipy scikit-image matplotlib pandas

REM Install basic I/O (no compilation needed)
echo Installing image I/O packages...
pip install tifffile pillow mrcfile dask

REM Install web interface
echo Installing web interface...
pip install flask werkzeug

REM Install scikit-learn for tracking
echo Installing tracking support...
pip install scikit-learn

echo.
echo ========================================
echo Minimal Installation Complete!
echo ========================================
echo.
echo Installed packages support:
echo   ✓ Napari viewer with all widgets
echo   ✓ TIFF, PNG, JPEG, BMP, MRC files
echo   ✓ All image processing features
echo   ✓ Segmentation and tracking
echo   ✓ Web interface
echo.
echo NOT installed (require C++ build tools):
echo   ✗ aicsimageio (advanced formats: CZI, ND2, LIF)
echo   ✗ cellpose (AI segmentation)
echo   ✗ stardist (AI segmentation)
echo.
echo The program will show a warning about aicsimageio
echo but all core features will work!
echo.
echo To start: run start.bat
echo.
pause
