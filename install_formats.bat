@echo off
REM Install format-specific image readers (no Java/C++ compilation needed)

echo ========================================
echo Installing Format-Specific Image Readers
echo (No Java or C++ compilation required)
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

echo Installing format-specific readers...
echo.

REM Install Zeiss CZI support
echo [1/6] Installing Zeiss CZI support...
pip install czifile
if %errorlevel% equ 0 (
    echo ✓ czifile installed - Zeiss CZI files supported
) else (
    echo ✗ czifile installation failed
)
echo.

REM Install Leica LIF support
echo [2/6] Installing Leica LIF support...
pip install readlif
if %errorlevel% equ 0 (
    echo ✓ readlif installed - Leica LIF files supported
) else (
    echo ✗ readlif installation failed
)
echo.

REM Install enhanced TIFF support
echo [3/6] Installing enhanced TIFF support...
pip install imageio
if %errorlevel% equ 0 (
    echo ✓ imageio installed - Enhanced format support
) else (
    echo ✗ imageio installation failed
)
echo.

REM Install universal format support
echo [4/6] Installing universal format support...
pip install pims
if %errorlevel% equ 0 (
    echo ✓ pims installed - Universal format support
) else (
    echo ✗ pims installation failed
)
echo.

REM Try Nikon ND2 support (might fail on some systems)
echo [5/6] Installing Nikon ND2 support...
pip install nd2reader
if %errorlevel% equ 0 (
    echo ✓ nd2reader installed - Nikon ND2 files supported
) else (
    echo ✗ nd2reader installation failed (common on some systems)
)
echo.

REM Install OpenCV for additional format support
echo [6/8] Installing OpenCV for additional formats...
pip install opencv-python
if %errorlevel% equ 0 (
    echo ✓ opencv-python installed - Additional format support
) else (
    echo ✗ opencv-python installation failed
)
echo.

REM Install Zarr support (for large array storage)
echo [7/8] Installing Zarr support...
pip install zarr
if %errorlevel% equ 0 (
    echo ✓ zarr installed - Zarr array support
) else (
    echo ✗ zarr installation failed
)
echo.

REM Install HDF5/Imaris support
echo [8/8] Installing HDF5/Imaris support...
pip install h5py
if %errorlevel% equ 0 (
    echo ✓ h5py installed - Imaris .ims files supported
) else (
    echo ✗ h5py installation failed
)
echo.

echo ========================================
echo Installation Summary
echo ========================================
echo.

REM Test what's available
python -c "
import sys
formats = []

try:
    import tifffile
    formats.append('✓ TIFF files (tifffile)')
except: pass

try:
    import czifile
    formats.append('✓ Zeiss CZI files (czifile)')
except: pass

try:
    import readlif
    formats.append('✓ Leica LIF files (readlif)')
except: pass

try:
    import nd2reader
    formats.append('✓ Nikon ND2 files (nd2reader)')
except: pass

try:
    import imageio
    formats.append('✓ Common formats (imageio)')
except: pass

try:
    import pims
    formats.append('✓ Universal support (pims)')
except: pass

try:
    import mrcfile
    formats.append('✓ MRC files (mrcfile)')
except: pass

try:
    from PIL import Image
    formats.append('✓ Basic formats (pillow)')
except: pass

try:
    import zarr
    formats.append('✓ Zarr arrays (zarr)')
except: pass

try:
    import h5py
    formats.append('✓ Imaris .ims files (h5py)')
except: pass

if formats:
    print('Supported formats:')
    for fmt in formats:
        print('  ' + fmt)
else:
    print('No image readers available!')
    
print()
print('You can now open these file types in PyMaris!')
"

echo.
echo Ready to use! Run start.bat to launch PyMaris.
echo.
pause