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

REM Check if installation was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Standard installation failed. Trying alternative method...
    echo Installing from src directory directly...
    pip install -e ./src
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ERROR: Both installation methods failed.
        echo Please check the error messages above and ensure:
        echo 1. Python 3.8+ is installed
        echo 2. All files are present in the src directory
        echo 3. Internet connection is available for downloading packages
        goto :error
    )
)

echo.
echo Installation complete successfully!
echo.
echo To activate the environment, run: venv\Scripts\activate.bat
echo To start the application, run: start.bat
goto :end

:error
echo.
echo Installation failed. Please check the errors above.

:end
pause
