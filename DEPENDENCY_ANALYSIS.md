# Dependency Analysis Report - Scientific Image Analyzer

> Note: This is a historical analysis document. Current supported installation flow is unified via `install.bat` (Windows) and `requirements.txt`.

## Summary
I've analyzed the entire codebase and identified all dependencies used throughout the project. Here's what I found and what I've updated:

## Missing Dependencies Identified

### Previously Missing from requirements.txt:
1. **magicgui** - Used in `magicgui_analysis_widget.py` for creating GUI widgets
2. **Flask & Werkzeug** - Used in `scientific_analyzer.py` for web interface  
3. **requests** - Used in `test_batch_hca_features.py` and `ai_segmentation_widget.py`
4. **mrcfile** - Used for MRC file export in various modules
5. **opencv-python** - Mentioned as optional dependency for advanced image processing
6. **cellpose, stardist, csbdeep** - AI segmentation models in `ai_segmentation_widget.py`

### Widget-specific Dependencies:
- **analysis_widget.py**: matplotlib backends, scipy.stats
- **hca_widget.py**: pandas, advanced statistical functions
- **processing_widget.py**: scipy.ndimage, scipy.signal, skimage.restoration
- **ai_segmentation_widget.py**: cellpose, stardist, torch, segment-anything
- **tracking_widget.py**: Advanced tracking algorithms
- **filament_tracing_widget.py**: skimage.morphology, scipy.spatial

## Files Updated

### 1. requirements.txt
- Added missing core dependencies
- Included web framework packages
- Added AI/ML packages as optional but recommended
- Added additional scientific packages

### 2. src/setup.py  
- Updated install_requires list to match requirements.txt
- Added version constraints for better compatibility
- Included all core dependencies

### 3. Dependency File Consolidation

#### requirements.txt (single source of truth)
- Unified dependency list for the supported installation flow
- Used directly by `install.bat`
- Includes core app, napari stack, and format readers

#### Compatibility aliases
- `requirements-full.txt` and `requirements-minimal.txt` now point to `requirements.txt`
- Kept only for compatibility with older commands

## Dependency Categories

### REQUIRED (Core functionality):
- napari, PyQt6, qtpy, magicgui
- numpy, scipy, scikit-image, matplotlib, pandas
- aicsimageio, tifffile, imageio, Pillow, h5py, zarr, dask
- napari ecosystem packages

### OPTIONAL (Enhanced features):
- Flask/Werkzeug (web interface)
- cellpose/stardist (AI segmentation)
- mrcfile, opencv-python (advanced I/O)
- Additional file format readers

### TEST/DEVELOPMENT:
- requests (for testing)
- Various format-specific readers

## Installation Commands

### Standard Installation:
```bash
pip install -r requirements.txt
pip install -e .
```

### Windows Unified Installer:
```bash
install.bat
```

## Verification

The dependency analysis covered:
- All Python files in src/ directory
- All widget files in src/widgets/
- Test files and example scripts
- Import statements and try/except blocks for optional dependencies
- Documentation references to external packages

All identified dependencies are now properly listed in the updated requirements files.