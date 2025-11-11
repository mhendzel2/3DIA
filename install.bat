@echo off
REM PyMaris Scientific Image Analyzer - Installation Script
REM This script creates a virtual environment and installs all dependencies

echo ========================================
echo PyMaris Scientific Image Analyzer
echo Installation Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/6] Checking Python version...
python --version

REM Check Python version (requires 3.8+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.

REM Create virtual environment
echo [2/6] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Removing old venv...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Try running: python -m pip install --user virtualenv
    pause
    exit /b 1
)
echo Virtual environment created successfully.
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install core dependencies
echo [5/6] Installing core dependencies...
echo This may take several minutes...
echo.

REM Install scientific computing stack first (most reliable)
echo Installing scientific computing packages...
pip install numpy scipy scikit-image matplotlib pandas
if %errorlevel% neq 0 (
    echo ERROR: Failed to install core scientific packages
    echo These are required for the application to work
    pause
    exit /b 1
)

REM Install napari and essential packages
echo Installing napari and core GUI packages...
pip install napari[pyqt6] PyQt6
if %errorlevel% neq 0 (
    echo WARNING: PyQt6 installation failed, trying PyQt5...
    pip install napari[pyqt5] PyQt5
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install napari GUI
        echo Try installing Visual C++ Build Tools from:
        echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
        pause
        exit /b 1
    )
)

REM Install basic image I/O packages (no C++ compilation needed)
echo Installing basic image I/O packages...
pip install tifffile pillow mrcfile dask

REM Try to install aicsimageio (may fail due to lxml compilation issues)
echo Installing advanced image I/O (aicsimageio)...
echo Note: This may fail on systems without C++ build tools
pip install aicsimageio 2>nul
if %errorlevel% neq 0 (
    echo WARNING: aicsimageio could not be installed
    echo This is likely due to missing C++ build tools
    echo The program will work with basic file formats (TIFF, PNG, JPEG)
    echo.
    echo To enable advanced formats, install:
    echo 1. Microsoft C++ Build Tools from:
    echo    https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo 2. Then run: venv\Scripts\pip install aicsimageio
)

REM Install optional AI/ML packages (may fail on some systems)
echo Installing optional AI/ML packages...
echo Note: These may take a while and require additional system libraries
pip install cellpose stardist --no-deps 2>nul
if %errorlevel% neq 0 (
    echo WARNING: AI packages (cellpose/stardist) could not be installed
    echo The program will still work without these features
)

REM Install tracking packages
echo Installing tracking packages...
pip install scikit-learn 2>nul

REM Install Flask for web interface (optional)
echo Installing web interface components...
pip install flask werkzeug

echo.
echo [6/6] Installation complete!
echo.

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat 2>nul

echo ========================================
echo Installation Summary
echo ========================================
echo Virtual environment: %CD%\venv
echo Python location: %CD%\venv\Scripts\python.exe
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

pause
