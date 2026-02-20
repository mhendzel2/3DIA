@echo off
REM Quick Start - PyMaris Scientific Image Analyzer
REM Minimal output version of start.bat

if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Running installer...
    call install.bat
    if %errorlevel% neq 0 exit /b 1
)

call venv\Scripts\activate.bat
cd src
python main_napari.py
cd ..
