@echo off
REM Quick install for Imaris (.ims) and Zarr support

echo ========================================
echo Installing Imaris and Zarr Support
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Virtual environment not found
    echo Run install_minimal.bat first
    pause
    exit /b 1
)

echo [1/2] Installing HDF5 support (for Imaris .ims files)...
pip install h5py
if %errorlevel% equ 0 (
    echo ✓ h5py installed successfully
    echo   You can now open Imaris .ims files!
) else (
    echo ✗ h5py installation failed
    echo   Imaris files will not be supported
)
echo.

echo [2/2] Installing Zarr support (for large array storage)...
pip install zarr
if %errorlevel% equ 0 (
    echo ✓ zarr installed successfully
    echo   You can now open Zarr arrays!
) else (
    echo ✗ zarr installation failed
    echo   Zarr files will not be supported
)
echo.

echo ========================================
echo Testing Installation
echo ========================================
python -c "
try:
    import h5py
    print('✓ Imaris .ims files supported')
except:
    print('✗ Imaris .ims files NOT supported')

try:
    import zarr
    print('✓ Zarr arrays supported')
except:
    print('✗ Zarr arrays NOT supported')
"

echo.
echo Installation complete!
echo.
echo To open files:
echo   - Imaris: File ^> Open ^> select .ims file
echo   - Zarr: File ^> Open ^> select .zarr directory
echo.
pause
