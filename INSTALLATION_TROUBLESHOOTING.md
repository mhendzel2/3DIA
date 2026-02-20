# PyMaris Installation Troubleshooting

## Common Installation Issues and Solutions

### Issue 1: `lxml` Build Failed (C++ Compiler Error)

**Error Message:**
```
error: command 'C:\Program Files (x86)\Microsoft Visual Studio\...\cl.exe' failed with exit code 2
ERROR: Failed building wheel for lxml
```

**Cause:** The `lxml` package (required by `aicsimageio`) needs to be compiled from source on your system, but you don't have the necessary C++ build tools installed.

**Solutions (Choose One):**

#### Option A: Install C++ Build Tools (Recommended for Full Features)

1. **Download Microsoft C++ Build Tools:**
   - Visit: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Download "Build Tools for Visual Studio 2022" (or latest)

2. **Install with Desktop Development with C++:**
   - Run the installer
   - Select "Desktop development with C++"
   - Click Install (requires ~6GB disk space)

3. **Restart your computer**

4. **Run the installer again:**
   ```batch
   install.bat
   ```

#### Option B: Use Minimal Installation (No C++ Required)

Run the minimal installer that skips `aicsimageio`:

```batch
install_minimal.bat
```

**What you'll have:**
- ✅ Full Napari viewer
- ✅ All analysis widgets
- ✅ TIFF, PNG, JPEG, BMP, MRC file support
- ✅ Image processing and segmentation
- ✅ Cell tracking and filament tracing
- ❌ Advanced formats (CZI, ND2, LIF) - use aicsimageio alternative
- ❌ AI segmentation (Cellpose, StarDist)

#### Option C: Use Pre-compiled Wheels

Try installing from pre-compiled wheels:

```batch
venv\Scripts\activate.bat
pip install --only-binary=:all: lxml
pip install aicsimageio
```

If that fails:
```batch
pip install aicsimageio-no-bioformats
```

#### Option D: Skip aicsimageio Entirely

Manually install everything except aicsimageio:

```batch
venv\Scripts\activate.bat

# Core packages
pip install napari[pyqt6]
pip install numpy scipy scikit-image matplotlib pandas
pip install tifffile pillow mrcfile dask
pip install flask werkzeug scikit-learn

# Now run the application
python src\main_napari.py
```

You'll see a warning: `Warning: aicsimageio not available. Limited file format support.`

This is **not a critical error** - the program works fine for standard formats!

---

### Issue 2: PyQt6 Installation Failed

**Error:** `ERROR: Failed building wheel for PyQt6`

**Solution:** Use PyQt5 instead:

```batch
venv\Scripts\activate.bat
pip uninstall PyQt6
pip install PyQt5
pip install napari[pyqt5]
```

---

### Issue 3: PowerShell Execution Policy Error

**Error:** 
```
File venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

**Solutions:**

**Option A:** Use Command Prompt instead:
```batch
# Open Command Prompt (cmd.exe) instead of PowerShell
venv\Scripts\activate.bat
python src\main_napari.py
```

**Option B:** Allow PowerShell scripts:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Option C:** Bypass activation:
```batch
# Run directly with venv Python
venv\Scripts\python.exe src\main_napari.py
```

---

### Issue 4: Napari Won't Start - Qt Platform Plugin Error

**Error:** `This application failed to start because no Qt platform plugin could be initialized`

**Solutions:**

1. **Reinstall PyQt:**
   ```batch
   venv\Scripts\activate.bat
   pip uninstall PyQt6 PyQt5
   pip install PyQt6
   ```

2. **Try PyQt5:**
   ```batch
   pip install PyQt5
   ```

3. **Update graphics drivers:**
   - Visit your GPU manufacturer's website (NVIDIA, AMD, Intel)
   - Download and install latest drivers

4. **Set Qt platform manually:**
   ```batch
   set QT_QPA_PLATFORM=windows
   python src\main_napari.py
   ```

---

### Issue 5: Out of Memory During Installation

**Error:** `MemoryError` or system freezes during `pip install`

**Solutions:**

1. **Install packages one at a time:**
   ```batch
   venv\Scripts\activate.bat
   pip install numpy
   pip install scipy
   pip install scikit-image
   pip install napari[pyqt6]
   # ... continue individually
   ```

2. **Disable parallel builds:**
   ```batch
   set MAX_JOBS=1
   pip install numpy scipy scikit-image
   ```

3. **Increase virtual memory:**
   - Windows: Settings → System → About → Advanced system settings
   - Advanced → Performance Settings → Advanced → Virtual memory → Change
   - Set custom size: Initial = 4096 MB, Maximum = 8192 MB

---

### Issue 6: ModuleNotFoundError After Installation

**Error:** `ModuleNotFoundError: No module named 'napari'`

**Cause:** Virtual environment not activated

**Solution:**

```batch
# Always activate venv first
venv\Scripts\activate.bat

# Then run
python src\main_napari.py
```

Or use start.bat which handles activation:
```batch
start.bat
```

---

### Issue 7: Import Error for 'src' Module

**Error:** `ModuleNotFoundError: No module named 'src'`

**Cause:** Import path issue in widget files

**Solution:** This was fixed in `magicgui_analysis_widget.py`. If you still see this:

```batch
# Make sure you're in the project root directory
cd C:\Users\mjhen\Github\3DIA

# Then run
venv\Scripts\python.exe src\main_napari.py
```

---

## Quick Diagnosis

Run this command to check your installation:

```batch
venv\Scripts\python.exe -c "import napari; import numpy; import scipy; print('Installation OK')"
```

**If successful:** You'll see `Installation OK`

**If it fails:** Note which package fails and reinstall it:
```batch
venv\Scripts\activate.bat
pip install --force-reinstall <package-name>
```

---

## System Requirements Check

### Minimum Requirements:
- ✅ Windows 10 or later
- ✅ Python 3.8 or later
- ✅ 4GB RAM
- ✅ 2GB free disk space

### Recommended:
- ✅ Windows 10/11
- ✅ Python 3.9 or later
- ✅ 16GB RAM
- ✅ 10GB free disk space
- ✅ Dedicated GPU

### Check Your System:

```batch
# Python version
python --version

# Available memory (PowerShell)
systeminfo | findstr /C:"Total Physical Memory"

# Disk space
wmic logicaldisk get size,freespace,caption
```

---

## Still Having Issues?

1. **Delete everything and start fresh:**
   ```batch
   rmdir /s /q venv
   install_minimal.bat
   ```

2. **Check Python installation:**
   ```batch
   python --version
   python -m pip --version
   where python
   ```

3. **Use minimal installation:**
   - Provides core functionality
   - No C++ build tools needed
   - Works on any Windows system

4. **Run diagnostics:**
   ```batch
   venv\Scripts\python.exe -m pip check
   ```

---

## Platform-Specific Notes

### Windows 11
- Usually works without issues
- Modern Python wheels available
- C++ build tools optional

### Windows 10
- May need C++ build tools for some packages
- Update to latest Windows 10 version recommended

### Older Windows (7/8)
- Not officially supported
- May work with minimal installation
- Some packages may not have wheels

---

## Getting Help

If none of these solutions work:

1. **Collect information:**
   ```batch
   python --version
   pip list > installed_packages.txt
   ```

2. **Note the exact error message**

3. **Check if minimal installation works:**
   ```batch
   install_minimal.bat
   ```

4. **Report issue with:**
   - Python version
   - Windows version
   - Full error message
   - Which installation method you tried

---

## Working Configuration (Verified)

This configuration is known to work:

```
Windows 10/11
Python 3.9.13
napari 0.4.18
PyQt6 6.4.0
numpy 1.24.3
scipy 1.10.1
scikit-image 0.21.0
matplotlib 3.7.1
```

If all else fails, match these exact versions:

```batch
venv\Scripts\activate.bat
pip install napari[pyqt6]==0.4.18 PyQt6==6.4.0 numpy==1.24.3 scipy==1.10.1 scikit-image==0.21.0 matplotlib==3.7.1 pandas tifffile pillow
```
