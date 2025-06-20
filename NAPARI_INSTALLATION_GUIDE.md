# PyMaris Scientific Image Analyzer - Napari Installation Guide

## Overview
Complete instructions for installing and running the Napari desktop version of PyMaris Scientific Image Analyzer locally on your system.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (3.9-3.11 recommended)
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space
- **Graphics**: OpenGL 3.3+ support

### Recommended Requirements
- **Python**: 3.11.x
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for AI features)
- **Monitor**: 1920x1080 or higher resolution

## Installation Methods

### Method 1: Automatic Installation (Recommended)

1. **Download the Package**
   ```bash
   # Download PyMaris_Scientific_Image_Analyzer_v2.0.0_XXXXXX.zip
   # Extract to your desired location
   unzip PyMaris_Scientific_Image_Analyzer_v2.0.0_*.zip
   cd PyMaris_Scientific_Image_Analyzer_v2.0.0_*
   ```

2. **Run Automatic Installer**
   ```bash
   # Windows
   python scripts/install.py
   # or double-click install.bat
   
   # macOS/Linux
   python3 scripts/install.py
   # or run ./scripts/install.sh
   ```

3. **Launch Napari Version**
   ```bash
   # Start Napari interface
   python main_napari.py
   
   # Or use the main launcher
   python main.py --mode napari
   ```

### Method 2: Manual Installation

1. **Install Python Dependencies**
   ```bash
   # Core Napari installation
   pip install napari[all]
   
   # Scientific computing stack
   pip install numpy scipy scikit-image matplotlib
   
   # Image I/O support
   pip install tifffile imageio aicsimageio
   
   # Additional microscopy formats
   pip install readlif pylibczirw czifile nd2reader
   
   # Optional AI features
   pip install cellpose stardist
   ```

2. **Install PyMaris Package**
   ```bash
   # Navigate to source directory
   cd src/
   
   # Install in development mode
   pip install -e .
   ```

3. **Verify Installation**
   ```bash
   python -c "import napari; print('Napari version:', napari.__version__)"
   python main_napari.py --test
   ```

## Running the Napari Application

### Basic Launch
```bash
# Standard Napari interface
python main_napari.py

# With debug output
python main_napari.py --debug

# Dual interface (Napari + Web)
python main.py --mode dual
```

### Command Line Options
```bash
python main_napari.py [OPTIONS]

Options:
  --debug          Enable debug logging
  --test           Run installation test
  --plugin-dir     Specify custom plugin directory
  --no-plugins     Disable plugin loading
  --help           Show help message
```

## Features Available in Napari Version

### Core Image Analysis
- **Multi-format Support**: CZI, LIF, ND2, TIFF, LSM, OIB, OIF
- **2D/3D/4D Viewing**: Time series and Z-stack navigation
- **Interactive Segmentation**: Manual and automated tools
- **Measurement Tools**: ROI analysis, object quantification

### Specialized Tools (Dock Widgets)
1. **Processing Widget**: Filtering, enhancement, preprocessing
2. **Segmentation Widget**: Cellpose, StarDist, watershed, thresholding
3. **Analysis Widget**: Object measurements, colocalization analysis
4. **FIB-SEM Tools**: 8 specialized plugins for electron microscopy
5. **Timelapse Tools**: 2D alignment, intensity normalization

### FIB-SEM Plugin Suite
- **3D Counter**: Volumetric object quantification
- **Tomo Slice Analyzer**: Tomographic slice viewing
- **APOC Classification**: GPU-accelerated pixel classification
- **Membrane Segmenter**: Specialized organelle segmentation
- **ChimeraX Integration**: 3D molecular visualization
- **Empanada Segmentation**: Deep learning EM segmentation
- **Organoid Counter**: 3D structure analysis
- **GPU Image Processor**: Accelerated processing

## Loading Image Data

### Supported Formats
- **Standard**: TIFF, PNG, JPEG, BMP
- **Microscopy**: CZI (Zeiss), LIF (Leica), ND2 (Nikon), LSM (Zeiss)
- **Electron Microscopy**: MRC, DM3, DM4
- **Other**: OIB/OIF (Olympus), STK, IMS

### Loading Methods
1. **File Menu**: File → Open Files → Select images
2. **Drag & Drop**: Drag files directly into Napari viewer
3. **Plugin Reader**: Automatic format detection
4. **Programmatic**: Using the file I/O widgets

### Timelapse Data Loading
```python
# For 2D timelapse sequences
# Name files: cells_t001.tif, cells_t002.tif, etc.
# Napari will automatically detect and load as stack

# For 3D timelapse (T, Z, Y, X)
# Name files: cells_t001_z001.tif, cells_t001_z002.tif, etc.
# Use the timelapse widget to organize properly
```

## Using the Analysis Widgets

### 1. Processing Widget
```python
# Access via: Plugins → PyMaris → Processing Tools
- Gaussian filtering
- Bilateral filtering (edge-preserving)
- Median filtering
- Gradient calculations
- Histogram equalization
```

### 2. Segmentation Widget
```python
# Access via: Plugins → PyMaris → Segmentation Tools
- Cellpose cell segmentation
- StarDist nucleus detection
- Watershed segmentation
- Threshold-based segmentation
- Manual region growing
```

### 3. Analysis Widget
```python
# Access via: Plugins → PyMaris → Analysis Tools
- Object measurements (area, perimeter, circularity)
- Intensity statistics
- Colocalization analysis
- Export to CSV/Excel
```

### 4. FIB-SEM Widget
```python
# Access via: Plugins → PyMaris → FIB-SEM Tools
- 3D object counting
- Membrane segmentation
- ChimeraX export
- GPU-accelerated processing
```

## Project Management

### Saving Projects
```python
# Save current state
File → Save Project → Select location

# Includes:
- All loaded images
- Analysis results
- Widget configurations
- Layer properties
```

### Loading Projects
```python
# Load saved project
File → Open Project → Select .napari file

# Restores:
- Image data and metadata
- Analysis layers
- Widget states
```

## Exporting Results

### Available Export Formats
- **Images**: TIFF, PNG, JPEG
- **Segmentation**: Label TIFF, CSV coordinates
- **Measurements**: CSV, Excel, JSON
- **Tracking**: TrackMate XML, CSV trajectories
- **3D Data**: MRC for ChimeraX

### Export Methods
1. **Right-click layers**: Save selected layer
2. **Analysis widget**: Export measurements
3. **File menu**: Export all or selected data
4. **Command palette**: Quick export commands

## Troubleshooting

### Common Issues

**1. Napari won't start**
```bash
# Check OpenGL support
python -c "import OpenGL; print('OpenGL OK')"

# Try software rendering
export MESA_GL_VERSION_OVERRIDE=3.3
python main_napari.py
```

**2. Plugin loading errors**
```bash
# Check plugin installation
napari --info

# Reinstall plugins
pip uninstall napari
pip install napari[all]
```

**3. File format not supported**
```bash
# Install additional readers
pip install aicsimageio[all]
pip install tifffile imageio-ffmpeg
```

**4. Performance issues**
```bash
# Enable GPU acceleration (if available)
pip install cupy-cuda11x  # or appropriate CUDA version

# Reduce image size for testing
# Use downsampling in viewer settings
```

### System-Specific Issues

**Windows**
- Install Visual C++ Redistributable
- Use conda environment for complex dependencies
- Check Windows Defender exclusions

**macOS**
- Install Xcode Command Line Tools
- Use Homebrew for system dependencies
- Check Gatekeeper settings for unsigned apps

**Linux**
- Install OpenGL development packages
- Check graphics driver installation
- Verify X11 forwarding for remote use

## Performance Optimization

### Memory Management
```python
# For large datasets
- Use memory mapping for TIFF files
- Process images in chunks
- Close unused layers
- Monitor memory usage in task manager
```

### GPU Acceleration
```python
# Enable CUDA support
pip install cupy-cuda11x
pip install cucim

# GPU-accelerated operations available in:
- APOC classification
- Some segmentation algorithms
- Large image processing
```

### Multi-threading
```python
# Napari automatically uses multiple cores for:
- Image rendering
- Some processing operations
- File I/O operations
```

## Advanced Usage

### Custom Plugins
```python
# Create custom analysis widgets
# See napari plugin development guide
# Use magicgui for quick GUI creation
```

### Batch Processing
```python
# Use the batch processor widget
# Or write scripts using napari headless mode
import napari
viewer = napari.Viewer(show=False)
# Process multiple files programmatically
```

### Integration with Other Tools
```python
# ImageJ integration
- Export/import ImageJ ROI files
- Use ImageJ macros via scripting

# MATLAB integration
- Export data as MAT files
- Use MATLAB Engine for Python

# Python ecosystem
- Integrate with pandas, seaborn
- Use scikit-learn for machine learning
- Connect to Jupyter notebooks
```

## Getting Help

### Documentation
- **Online Docs**: Check replit.md for latest updates
- **Napari Docs**: https://napari.org/
- **Examples**: See examples/ directory in package

### Support
- **Issue Tracking**: Check TROUBLESHOOTING.md
- **Community**: Napari community forums
- **Debug Mode**: Run with --debug for detailed logs

### Reporting Issues
Include in bug reports:
- Operating system and version
- Python and Napari versions
- Complete error messages
- Steps to reproduce
- Sample data (if possible)

## Next Steps

After installation:
1. **Test with sample data**: Use provided examples
2. **Explore widgets**: Try each analysis tool
3. **Load your data**: Start with small test images
4. **Save workflows**: Document successful analysis pipelines
5. **Customize interface**: Arrange widgets for your workflow

The Napari version provides the most comprehensive analysis capabilities with full access to all plugins and advanced features.