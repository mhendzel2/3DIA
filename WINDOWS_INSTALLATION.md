# PyMaris Installation and Startup Guide (Windows)

## Quick Start

### First Time Setup - Choose Your Installation Method

#### Method 1: Full Installation (Recommended)
- Requires C++ Build Tools for advanced file formats
- Supports CZI, ND2, LIF files via aicsimageio

```batch
install.bat
```

#### Method 2: Minimal Installation (No C++ Required)
- Works on any Windows system
- Supports TIFF, PNG, JPEG, BMP, MRC files
- No compilation needed

```batch
install_minimal.bat
```

**Having installation issues?** See `INSTALLATION_TROUBLESHOOTING.md` for solutions to common problems.

### Starting the Application

After installation completes:
```batch
start.bat
```

Or for quick launch (auto-installs if needed):
```batch
quickstart.bat
```

## Installation Methods Comparison

| Feature | Full Install | Minimal Install |
|---------|-------------|-----------------|
| C++ Build Tools Required | Yes | No |
| Installation Time | 10-20 min | 5-10 min |
| TIFF, PNG, JPEG, BMP | ✅ | ✅ |
| CZI, ND2, LIF files | ✅ | ❌ |
| All Analysis Features | ✅ | ✅ |
| AI Segmentation | Optional | ❌ |
| Disk Space | ~3GB | ~2GB |

## Detailed Instructions

### Prerequisites

- **Windows 10 or 11**
- **Python 3.8 or higher** - Download from [python.org](https://www.python.org/)
  - ⚠️ **Important:** Check "Add Python to PATH" during installation
- **At least 4GB RAM** (8GB+ recommended for large datasets)
- **2GB free disk space** for dependencies

### For Full Installation (Optional):
- **Microsoft C++ Build Tools** - Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Required for `aicsimageio` (advanced file format support)
  - Select "Desktop development with C++" during installation
  - ~6GB disk space required

### Installation Process

The `install.bat` script performs the following steps:

1. ✅ Checks if Python is installed and accessible
2. ✅ Creates a virtual environment in the `venv` folder
3. ✅ Upgrades pip to the latest version
4. ✅ Installs Napari with Qt support
5. ✅ Installs scientific computing packages (numpy, scipy, scikit-image)
6. ✅ Installs image I/O libraries (tifffile, aicsimageio, mrcfile)
7. ✅ Attempts to install optional AI/ML packages (cellpose, stardist)
8. ✅ Installs Flask for the web interface

**Total installation time:** 5-15 minutes depending on internet speed

### Starting the Application

The `start.bat` script:

1. ✅ Activates the virtual environment
2. ✅ Launches the Napari desktop interface
3. ✅ Displays available features and widgets

**Startup time:** 5-30 seconds depending on system

## Troubleshooting

### Problem: "Python is not installed or not in PATH"

**Solution:**
1. Install Python from [python.org](https://www.python.org/)
2. During installation, check ☑️ "Add Python to PATH"
3. Restart your computer
4. Run `install.bat` again

**Alternative:** Add Python to PATH manually
- Search for "Environment Variables" in Windows
- Add Python installation directory to PATH

### Problem: "Virtual environment activation failed"

**Solution:**
```batch
# Delete the venv folder and try again
rmdir /s /q venv
install.bat
```

### Problem: "Import errors" when starting

**Solution:**
```batch
# Reinstall dependencies
venv\Scripts\activate.bat
pip install --upgrade --force-reinstall napari[all] PyQt6
```

### Problem: "Qt platform plugin could not be initialized"

**Solutions:**
1. Try PyQt5 instead:
   ```batch
   venv\Scripts\activate.bat
   pip uninstall PyQt6
   pip install PyQt5
   ```

2. Update graphics drivers

3. Run with environment variable:
   ```batch
   set QT_QPA_PLATFORM=windows
   start.bat
   ```

### Problem: Application is slow or crashes with large images

**Solutions:**
- Increase virtual memory (pagefile) in Windows
- Close other applications to free RAM
- Use Volume Clipping to reduce data size
- Process in smaller chunks

## Manual Installation (Alternative)

If the batch files don't work, install manually:

```batch
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate.bat

# 3. Install dependencies
pip install napari[all] PyQt6
pip install numpy scipy scikit-image matplotlib pandas
pip install aicsimageio tifffile mrcfile pillow dask
pip install flask werkzeug

# 4. Optional: AI packages
pip install cellpose stardist

# 5. Run the application
cd src
python main_napari.py
```

## Command Line Options

### Running Specific Interfaces

**Napari Desktop (default):**
```batch
start.bat
```

**Web Interface:**
```batch
venv\Scripts\activate.bat
python src\main.py enhanced
```

**Simple Analyzer:**
```batch
venv\Scripts\activate.bat
python src\main.py web
```

## Updating the Application

To update to the latest version:

```batch
# 1. Pull latest code (if using git)
git pull

# 2. Update dependencies
venv\Scripts\activate.bat
pip install --upgrade napari numpy scipy scikit-image

# 3. Restart application
start.bat
```

## Uninstallation

To completely remove PyMaris:

1. Delete the `venv` folder
2. Delete the application folder
3. (Optional) Uninstall Python if not needed for other projects

## File Structure

```
3DIA/
├── install.bat          # Installation script (run first)
├── start.bat            # Startup script (run to launch)
├── venv/                # Virtual environment (created by install.bat)
│   ├── Scripts/
│   │   ├── python.exe
│   │   └── activate.bat
│   └── Lib/
├── src/
│   ├── main_napari.py   # Napari interface entry point
│   ├── main.py          # Multi-interface launcher
│   └── widgets/         # All analysis widgets
└── config/
    └── config.json      # Configuration file
```

## Performance Tips

### For Large Datasets (>2GB)

1. **Use Volume Clipping** - Reduce data size before processing
2. **Enable Memory Mapping** - In File I/O widget
3. **Process in Chunks** - Use batch processing for multiple files
4. **Increase Virtual Memory:**
   - Settings → System → About → Advanced system settings
   - Performance → Settings → Advanced → Virtual memory
   - Set custom size: 1.5x to 3x your RAM

### For Faster Tracking

1. Reduce `max_distance` parameter
2. Disable gap closing for initial analysis
3. Filter short tracks after completion
4. Use fewer timepoints for testing

### For Better Rendering

1. Start with low sampling rate, increase for final render
2. Use MIP for quick previews
3. Enable GPU acceleration (if available in future versions)
4. Close unused widgets to free memory

## Support

- **Documentation:** See `FEATURES_AND_USAGE.md`
- **GitHub Issues:** https://github.com/mhendzel2/3DIA/issues
- **Installation Guide:** `NAPARI_INSTALLATION_GUIDE.md`

## System Requirements

### Minimum
- Windows 10
- Python 3.8+
- 4GB RAM
- 2GB disk space
- Integrated graphics

### Recommended
- Windows 10/11
- Python 3.9+
- 16GB RAM
- 10GB disk space
- Dedicated GPU (for future features)
- SSD for faster loading

## What's Included

The installation includes these major features:

- ✅ **Volume Rendering** - MIP, alpha blending, orthogonal views
- ✅ **Filament Tracing** - Neuron and cytoskeleton analysis
- ✅ **Cell Tracking** - With lineage trees and division detection
- ✅ **Segmentation** - Multiple algorithms including AI-based
- ✅ **Colocalization** - Statistical channel overlap analysis
- ✅ **3D Surface Rendering** - Marching cubes isosurface
- ✅ **Spot Detection** - Blob detection in 2D/3D
- ✅ **Image Processing** - Filtering, thresholding, morphology
- ✅ **Statistics** - Comprehensive object measurements
- ✅ **Export** - Multiple formats for compatibility

## License

MIT License - Free for academic and commercial use
