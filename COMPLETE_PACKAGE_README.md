# PyMaris Scientific Image Analyzer v2.0.0 - Complete Package

## Package Contents

**Package Name**: `PyMaris_Scientific_Image_Analyzer_v2.0.0_20250619_171412.zip`
**Size**: 0.1 MB
**Build Date**: June 19, 2025

## What's Included

### Core Application Files
- **Dual Interface**: Napari desktop + enhanced web interface
- **Multi-file Timelapse**: 2D and 3D sequence support with automatic filename parsing
- **TIF Loading Fix**: Comprehensive fallback TIFF reader for 8-bit and 16-bit files
- **FIB-SEM Tools**: 8 specialized electron microscopy analysis plugins
- **2D Alignment**: Cross-correlation, feature-based, and mutual information methods
- **Export Integration**: Multiple formats for tracking software compatibility

### Installation & Setup
- **Cross-platform installers**: Windows `.bat`, Unix `.sh`, Python installer
- **Quick start script**: `quick_start_napari.py` for simplified Napari launch
- **Automatic dependency detection**: System diagnostic and installation tools
- **Configuration files**: Customizable settings for performance and features

### Documentation
- **Complete installation guide**: `NAPARI_INSTALLATION_GUIDE.md`
- **User manual**: Step-by-step usage instructions
- **API reference**: Developer documentation for extensions
- **Troubleshooting guide**: Common issues and solutions

### Debugging Tools
- **System diagnostic**: `debug/diagnostic.py` - comprehensive system analysis
- **Performance benchmark**: `debug/benchmark.py` - speed and memory testing
- **TIF file analyzer**: `tif_diagnostic.py` - TIFF file structure analysis
- **Detailed logging**: Configurable debug output for issue tracking

### Example Data & Workflows
- **Pre-configured workflows**: Cell analysis, FIB-SEM, timelapse processing
- **Test data generators**: Create sample images for testing
- **Template configurations**: Ready-to-use analysis pipelines

## Key Features

### Multi-file Timelapse Support (NEW)
- **Batch Upload**: `/api/upload/batch` endpoint for multiple files
- **Automatic Parsing**: Detects timepoint and Z-slice from filenames
- **Sequence Organization**: Proper temporal and spatial ordering
- **Supported Patterns**: 
  - `cells_t001.tif`, `cells_t002.tif` (2D timelapse)
  - `cells_t001_z001.tif`, `cells_t001_z002.tif` (3D timelapse)
  - `sample_s1_t3.tif` (position and timepoint)

### TIF File Loading (FIXED)
- **Fallback TIFF Reader**: Works without external dependencies
- **16-bit Support**: Handles both 8-bit and 16-bit uncompressed TIFF
- **Comprehensive Parsing**: Reads TIFF headers and metadata
- **Error Recovery**: Graceful handling of corrupted or unsupported files

### Analysis Capabilities
- **Segmentation**: Cellpose, StarDist, watershed, thresholding
- **Measurements**: Area, perimeter, circularity, intensity statistics
- **Colocalization**: Channel overlap analysis with statistical measures
- **3D Analysis**: Volumetric object counting and morphometrics
- **Tracking**: Object trajectory analysis for time series

### Export Formats
- **Tracking Software**: TrackMate XML, CSV trajectories, centroids
- **ImageJ Compatible**: ROI files, label TIFF masks
- **Data Analysis**: CSV measurements, JSON metadata
- **3D Visualization**: MRC format for ChimeraX integration

## Installation Instructions

### Quick Start (Recommended)
1. Extract the package to your desired location
2. Run the quick start script:
   ```bash
   python quick_start_napari.py
   ```
3. Follow the prompts for automatic installation

### Manual Installation
1. Install Python 3.8+ if not already installed
2. Run the appropriate installer:
   - **Windows**: Double-click `install.bat`
   - **macOS/Linux**: Run `./scripts/install.sh`
   - **Any platform**: `python scripts/install.py`

### Napari Desktop Version
```bash
# Install Napari and dependencies
pip install napari[all] numpy scipy scikit-image matplotlib

# Launch the application
python main_napari.py
```

### Web Interface Only
```bash
# No additional dependencies required
python main.py --mode web
# Access at http://localhost:5000
```

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- OpenGL 3.3+ (for Napari)
- 2GB free storage

### Recommended
- Python 3.11.x
- 16GB RAM
- NVIDIA GPU with CUDA (for AI features)
- 1920x1080+ display

## Debugging and Support

### System Diagnostics
```bash
# Run comprehensive system check
python debug/diagnostic.py

# Performance benchmark
python debug/benchmark.py

# Analyze specific TIF files
python tif_diagnostic.py
```

### Common Issues
1. **TIF loading fails**: Check file with `tif_diagnostic.py`
2. **Napari won't start**: Verify OpenGL support
3. **Performance issues**: Run benchmark, check memory usage
4. **Plugin errors**: Check diagnostic report for missing dependencies

### Log Files
- Application logs: `logs/analyzer.log`
- Debug output: Enable with `--debug` flag
- Error reports: Automatically generated in debug mode

## Usage Examples

### Loading Timelapse Sequences
```python
# Web interface: Use batch upload with multiple files
# Napari: Drag and drop multiple files, or use File → Open Files

# Supported naming conventions:
# - cells_t001.tif, cells_t002.tif (time series)
# - data_t01_z05.tif (time and Z-stack)
# - sample_s1_t3.tif (position and time)
```

### Running Analysis Workflows
```bash
# Start with example data
python examples/generate_test_data.py

# Process with pre-configured workflow
# Use Napari widgets or web interface controls
```

### Exporting for External Software
- **ImageJ**: Export ROI files and label masks
- **TrackMate**: Export tracking-ready CSV files
- **MATLAB**: Export MAT files with measurements
- **ChimeraX**: Export 3D data in MRC format

## What's New in v2.0.0

### June 19, 2025 Updates
- ✓ Multi-file timelapse loading with automatic sequence detection
- ✓ Fixed TIF file loading with comprehensive fallback reader
- ✓ Enhanced API endpoints for batch processing
- ✓ Improved filename parsing for microscopy naming conventions
- ✓ Better error handling and diagnostic tools

### June 16, 2025 Updates
- ✓ 2D image alignment with 4 different algorithms
- ✓ Timelapse intensity normalization (6 methods)
- ✓ Comprehensive package creation with debugging tools

### June 14, 2025 Updates
- ✓ 8 specialized FIB-SEM analysis plugins
- ✓ ChimeraX integration for 3D visualization
- ✓ Export system for tracking software integration

## Support and Documentation

### Included Documentation
- `NAPARI_INSTALLATION_GUIDE.md`: Complete Napari setup
- `docs/USER_GUIDE.md`: Comprehensive usage manual
- `docs/TROUBLESHOOTING.md`: Problem solving guide
- `docs/API_REFERENCE.md`: Developer documentation

### Getting Help
1. Check the troubleshooting guide for common issues
2. Run diagnostic tools to identify system problems
3. Review log files for detailed error messages
4. Test with provided example data to verify installation

## License and Distribution

- **License**: MIT License (see LICENSE file)
- **Distribution**: Complete standalone package
- **Dependencies**: Automatically managed by installers
- **Compatibility**: Windows, macOS, Linux

The package is ready for immediate deployment and use on any compatible system. All core functionality works without external dependencies, with optional enhancements available through the installation scripts.