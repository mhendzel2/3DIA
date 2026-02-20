@echo off
REM PyMaris Scientific Image Analyzer - Startup Script
REM This script activates the virtual environment and launches the Napari interface

echo ========================================
echo PyMaris Scientific Image Analyzer
echo Napari Desktop Interface
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    echo Try reinstalling by running install.bat
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Check if main_napari.py exists
if not exist "src\main_napari.py" (
    echo ERROR: main_napari.py not found in src directory
    echo Please ensure the application files are in the correct location.
    pause
    exit /b 1
)

REM Launch the Napari interface
echo [2/2] Starting PyMaris Napari Interface...
echo.
echo ========================================
echo Application Starting...
echo ========================================
echo.
echo New Imaris-like features available:
echo   - Volume Rendering (MIP, Alpha Blending, Orthogonal Views, Clipping)
echo   - Filament Tracing (Neuron/Cytoskeleton Analysis)
echo   - Advanced Cell Tracking (Lineage Trees, Gap Closing, Division Detection)
echo.
echo The Napari window will open shortly...
echo Close this window to exit the application.
echo ========================================
echo.

REM Change to src directory and run the application
cd src
python main_napari.py

REM Check if application exited with error
if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo ERROR: Application exited with an error
    echo Error code: %errorlevel%
    echo ========================================
    echo.
    echo Troubleshooting tips:
    echo 1. Try reinstalling: run install.bat
    echo 2. Check if all dependencies are installed
    echo 3. Make sure you have Qt libraries installed
    echo 4. Try updating graphics drivers
    echo.
    pause
    exit /b %errorlevel%
)

REM Return to root directory
cd ..

echo.
echo ========================================
echo Application closed successfully
echo ========================================
echo.
