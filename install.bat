@echo off

echo Creating virtual environment...
REM Create a virtual environment
python -m venv venv

echo Activating virtual environment...
REM Activate the virtual environment
call venv\Scripts\activate.bat

echo Upgrading pip...
REM Upgrade pip to latest version
python -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
REM Install dependencies from requirements file
pip install -r requirements.txt

echo Installation complete.
echo.
echo To activate the environment, run: venv\Scripts\activate.bat
echo To start the application, run: start.bat
pause
