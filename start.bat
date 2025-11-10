@echo off

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Run the main napari application
python src/main_napari.py

REM Deactivate the virtual environment
deactivate
