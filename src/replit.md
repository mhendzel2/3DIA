# replit.md

## Overview

This is a Scientific Image Analysis application that replicates Napari Hub plugin functionality in a web-based interface. The application provides comprehensive microscopy analysis tools including segmentation, object tracking, colocalization analysis, and statistical measurements. It's designed to work both as a standalone web application and as a Napari-based plugin suite.

## System Architecture

### Frontend Architecture
- **Web Interface**: HTML5/CSS3/JavaScript with Canvas-based image visualization
- **Interactive Controls**: Real-time analysis controls with responsive design
- **Visualization**: Matplotlib-based plotting with web integration
- **User Experience**: Modern dashboard with tabbed interface for different analysis modes

### Backend Architecture
- **Web Framework**: Flask-based HTTP server with RESTful API endpoints
- **Core Engine**: Python-based scientific computing with fallback implementations
- **Plugin System**: Napari-compatible widget architecture using magicgui
- **Processing Pipeline**: Modular analysis components for different microscopy techniques

## Key Components

### Image Processing Engine
- **Segmentation Algorithms**: 
  - Cellpose-inspired classical computer vision implementation
  - StarDist-style star-convex object detection
  - Connected component labeling and boundary detection
- **Filtering Operations**: Gaussian blur, median filtering, gradient calculations
- **Object Detection**: Local maxima detection, blob detection algorithms

### Analysis Tools
- **Region Properties**: Morphological measurements (area, perimeter, circularity, aspect ratio)
- **Colocalization Analysis**: Statistical correlation coefficients (Pearson, Manders overlap)
- **Object Tracking**: Multi-frame trajectory analysis with distance-based association
- **Statistical Analysis**: Intensity-based statistics and spatial distribution analysis

### Web Application Components
- **Main Entry Point**: `main.py` - Simple launcher for the web application
- **Core Application**: `simple_analyzer.py` - Standalone web server with scientific analysis
- **Enhanced Features**: `scientific_analyzer.py` - Extended functionality with additional algorithms
- **Flask Application**: `app.py` - Full-featured web framework implementation

### Napari Integration
- **Plugin Widgets**: Modular widget system for processing, segmentation, and analysis
- **File I/O Support**: Multi-format microscopy file reader with metadata handling
- **Visualization**: 3D rendering controls and advanced visualization options
- **Project Management**: Session state management and workflow persistence

## Data Flow

1. **Image Input**: 
   - Web upload interface or Napari file dialog
   - Support for multiple microscopy formats (CZI, LIF, ND2, TIFF, etc.)
   - Automatic format detection and metadata extraction

2. **Processing Pipeline**:
   - Image preprocessing (filtering, enhancement)
   - Segmentation algorithm selection and parameter tuning
   - Object detection and measurement
   - Statistical analysis and visualization

3. **Results Output**:
   - Interactive web visualization
   - Statistical summaries and plots
   - Export capabilities for data and images
   - Project saving and session management

## External Dependencies

### Core Dependencies
- **Optional Scientific Stack**: NumPy, SciPy, scikit-image, matplotlib (with fallbacks)
- **Web Framework**: Flask, Werkzeug (optional - standalone mode available)
- **Image Processing**: PIL/Pillow (optional - basic implementations provided)

### Napari Integration Dependencies
- **Napari Framework**: napari[all], magicgui, QtPy
- **Advanced Plugins**: cellpose, stardist, btrack (optional)
- **File I/O**: aicsimageio, tifffile, imageio (optional)

### Fallback Strategy
The application is designed to work with pure Python without external dependencies. All core functionality has fallback implementations when scientific libraries are not available.

## Deployment Strategy

### Web Application
- **Standalone Mode**: Pure Python HTTP server with built-in analysis tools
- **Flask Mode**: Full web framework with enhanced features and API endpoints
- **Port Configuration**: Default port 5000 with configurable external access

### Napari Plugin
- **Plugin Manifest**: `napari.yaml` defines readers, writers, and widgets
- **Widget Integration**: Dock-able widgets integrated into Napari interface
- **File Format Support**: Comprehensive microscopy format reader implementation

### Package Distribution
- **Deployment Scripts**: Automated package creation and distribution tools
- **Installation Support**: Dependency checking and automatic installation
- **Cross-Platform**: Support for Windows, macOS, and Linux environments

## Recent Changes

### June 20, 2025 - Comprehensive Debugging and Final Package Release
- **Fixed Duplicate Widget Files**: Created main_napari_fixed.py with proper advanced widget imports and fallback logic
- **Implemented Proper Phase Correlation**: Replaced placeholder with FFT-based scipy implementation for accurate timelapse alignment
- **Corrected Misleading Class Names**: Renamed CellposeSegmentation to GradientWatershedSegmentation with clear algorithm documentation
- **Enhanced ChimeraX Integration**: Robust path detection and standard MRC export via configuration system
- **Resolved All Dependency Issues**: Comprehensive fallbacks for OpenCV, pandas, scipy with pure Python alternatives
- **Released Final Debugged Package**: PyMaris_Scientific_Image_Analyzer_FINAL_DEBUGGED_v2.0.0_20250620_143437.zip (0.20 MB)

### June 16, 2025 - Multi-file Timelapse Support and TIF Loading Fixes
- **Fixed TIF File Loading**: Implemented comprehensive fallback TIFF reader supporting 8-bit and 16-bit uncompressed files
- **Multi-file Timelapse Loading**: Added batch upload capabilities for 2D and 3D timelapse sequences
- **Filename Parsing**: Automatic detection of timepoint and Z-slice information from microscopy naming conventions
- **Sequence Organization**: Support for organizing files into proper temporal and spatial sequences
- **Enhanced API Endpoints**: Added `/api/upload/batch` and `/api/timelapse/load` for sequence handling

### June 14, 2025 - FIB-SEM Specialized Tools Integration
- **ChimeraX Integration**: Added ChimeraXIntegration class for 3D molecular visualization and FIB-SEM correlation
- **8 Specialized FIB-SEM Tools Implemented**:
  1. **3D Counter**: Advanced 3D object counting and volume analysis for volumetric data
  2. **Tomo Slice Analyzer**: Advanced tomographic slice viewing and orthogonal navigation
  3. **Accelerated Classification (APOC)**: GPU-accelerated pixel and object classification with texture feature extraction
  4. **Membrane Segmenter**: Specialized segmentation for membrane-bound structures (vesicles, organelles)
  5. **GPU Image Processor**: Accelerated image processing for large FIB-SEM datasets
  6. **Empanada Segmentation**: Deep learning-based segmentation for EM data with pre-trained models
  7. **Organoid Counter**: 3D structure analysis and quantification with morphometrics
  8. **FIB-SEM Analyzer**: Main integration class combining all specialized tools

### Export System Enhancement
- **Tracking Software Integration**: Added comprehensive export functionality for single particle tracking software
- **Multiple Export Formats**: TIFF masks, CSV coordinates, tracking-ready formats, centroids, measurements, ImageJ ROI
- **Real-time Download Links**: Direct download capability for exported analysis data

### Web Interface Enhancements  
- **FIB-SEM Control Panel**: Added specialized UI controls for all 8 FIB-SEM tools
- **Advanced API Endpoints**: Integrated `/api/fibsem/*` endpoints for specialized analysis workflows
- **Fallback Implementation**: Graceful degradation when FIB-SEM plugins unavailable

## Changelog

- June 13, 2025. Initial setup
- June 14, 2025. ChimeraX integration and 8 FIB-SEM specialized tools added

## User Preferences

Preferred communication style: Simple, everyday language.