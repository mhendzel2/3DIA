@echo off

echo Fixing PyQt6 dependency issue...
echo.

REM Activate the virtual environment
call venv\Scripts\activate.bat

echo Installing PyQt6...
pip install PyQt6>=6.4.0 PyQt6-Qt6>=6.4.0 qtpy>=2.3.0

echo Installing missing napari packages...
pip install console>=0.1.4 napari-plugin-engine>=0.2.0 napari-svg>=0.2.1 npe2>=0.7.9 numpydoc>=1.9.0 ome-types>=0.6.2 ome-zarr>=0.10.2

echo.
echo PyQt6 and dependencies installed successfully!
echo You can now run start.bat to launch the application.
pause