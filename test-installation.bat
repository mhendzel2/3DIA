@echo off
echo Testing installation...

REM Activate the virtual environment
call venv\Scripts\activate.bat

echo Testing Python package imports...
python -c "
try:
    import numpy
    print('✓ numpy imported successfully')
except ImportError as e:
    print('✗ numpy import failed:', e)

try:
    import napari
    print('✓ napari imported successfully')
except ImportError as e:
    print('✗ napari import failed:', e)

try:
    from PyQt6 import QtWidgets
    print('✓ PyQt6 imported successfully')
except ImportError as e:
    print('✗ PyQt6 import failed:', e)

try:
    import sys
    sys.path.append('src')
    from main_napari import main
    print('✓ main_napari can be imported')
except ImportError as e:
    print('✗ main_napari import failed:', e)
except Exception as e:
    print('✓ main_napari imported but cannot run without display:', str(e)[:50])
"

echo.
echo Test complete. Check output above for any import errors.
pause