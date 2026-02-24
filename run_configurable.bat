@echo off
REM Run PyMaris with Configurable Widget Loading
REM This version allows you to choose which widgets load at startup

echo ========================================
echo PyMaris - Configurable Widget Mode
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first.
    echo.
    pause
    exit /b 1
)

echo Starting PyMaris with configurable widgets...
echo.
echo To configure widgets:
echo   1. Use the Widget Manager dock widget
echo   2. Or edit config/widget_config.json
echo Optional launch flags:
echo   --workspace tracking
echo   --workspace high_content_screening
echo   --workspace viz_3d_quant
echo.

REM Activate venv and run
call venv\Scripts\activate.bat
python src\main_napari_configurable.py %*

pause
