@echo off

REM Create a virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install the package and its dependencies
pip install -e src

REM Deactivate the virtual environment
deactivate

echo Installation complete.
pause
