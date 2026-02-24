@echo off
setlocal
set NO_PAUSE=0
if /I "%~1"=="--no-pause" set NO_PAUSE=1
REM PyMaris Scientific Image Analyzer - Unified Installation Script
REM Single installer for core app + formats + Imaris/Zarr + dependency repair checks

echo ========================================
echo PyMaris Scientific Image Analyzer
echo Unified Installation Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo [1/8] Checking Python version...
python --version

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if %errorlevel% neq 0 (
    echo ERROR: Python 3.10+ is required by this project.
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

REM Check Python version (requires 3.10+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.

REM Create virtual environment
echo [2/8] Preparing virtual environment...
if not exist "venv\Scripts\python.exe" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        echo Try running: python -m pip install --user virtualenv
        if "%NO_PAUSE%"=="0" pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Existing virtual environment found. Reusing venv.
)
echo.

REM Activate virtual environment
echo [3/8] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Upgrade pip
echo [4/8] Upgrading pip and build tools...
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
echo.

REM Install dependency baseline
echo [5/8] Installing unified dependency baseline from requirements.txt...
echo This may take several minutes...
echo.
python -m pip install --prefer-binary -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Full baseline install failed. Attempting fallback install path...
    echo.
    echo Installing required core packages...
    python -m pip install --prefer-binary numpy scipy scikit-image pandas matplotlib seaborn scikit-learn imageio Pillow tifffile h5py zarr "dask[array]" mrcfile napari PyQt6 PyQt6-Qt6 qtpy magicgui napari-plugin-engine napari-svg npe2 Flask Werkzeug requests
    if %errorlevel% neq 0 (
        echo ERROR: Core fallback dependency installation failed.
        echo Run INSTALLATION_TROUBLESHOOTING.md for detailed recovery steps.
        if "%NO_PAUSE%"=="0" pause
        exit /b 1
    )

    echo.
    echo Installing extended format/tooling packages ^(best effort^)...
    python -m pip install --prefer-binary ome-types ome-zarr opencv-python pims readlif nd2reader czifile numpydoc console
    if %errorlevel% neq 0 (
        echo WARNING: Some extended packages failed to install. Core install is still usable.
    )
)

echo.
echo [6/8] Running dependency repair pass (legacy fix-dependencies compatibility)...
python -m pip install --prefer-binary PyQt6 PyQt6-Qt6 qtpy
python -m pip install --prefer-binary console napari-plugin-engine napari-svg npe2 numpydoc ome-types ome-zarr
if %errorlevel% neq 0 (
    echo WARNING: Some repair-pass packages reported issues. Continuing...
)

echo.
echo [7/8] Installing project package...
python -m pip install -e .
if %errorlevel% neq 0 (
    echo ERROR: Failed to install local project package.
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo.
echo [8/8] Verifying key imports and format support...
python -c "import numpy, scipy, skimage, napari, tifffile, h5py, zarr, mrcfile, imageio, pandas, flask, requests, readlif, nd2reader, czifile, pims; print('Dependency verification OK')"
if %errorlevel% neq 0 (
    echo WARNING: One or more optional modules failed verification.
    echo Run test-installation.bat for detailed diagnostics.
)

echo.
echo Running pip consistency check...
python -m pip check
if %errorlevel% neq 0 (
    echo WARNING: pip check found version conflicts. See output above.
)

echo.
echo Installation complete!
echo.

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat 2>nul

echo ========================================
echo Installation Summary
echo ========================================
echo Virtual environment: %CD%\venv
echo Python location: %CD%\venv\Scripts\python.exe
echo.
echo Included in this install:
echo   - Core application dependencies
echo   - Napari + Qt stack
echo   - Imaris (.ims) via h5py
echo   - Zarr arrays via zarr
echo   - Format readers: CZI/LIF/ND2/pims/imageio/opencv
echo   - Legacy dependency repair pass
echo.
echo To start the application, run: start.bat
echo Or manually:
echo   1. venv\Scripts\activate.bat
echo   2. python src\main_napari.py
echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.

if "%NO_PAUSE%"=="0" pause
