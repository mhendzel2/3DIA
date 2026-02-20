# PyMaris - Complete Napari Plugin Functionality Overview

## 🎯 Overview

PyMaris is a comprehensive scientific image analysis platform built **on top of Napari**, extending it with Imaris-like capabilities. All functionality is accessible through custom dock widgets in the Napari viewer.

---

## 🏗️ Architecture

```
PyMaris Application
    ↓
Napari Viewer (Base Platform)
    ↓
Custom Dock Widgets (Our Added Functionality)
    ├── File I/O Widget
    ├── Processing Widget (842 lines)
    ├── Segmentation Widget (745 lines)
    ├── Analysis Widget
    ├── 3D Visualization Widget (153 lines)
    ├── Deconvolution Widget
    ├── Statistics Widget
    ├── Filament Tracing Widget (514 lines) ← NEW
    ├── Cell Tracking & Lineage Widget (684 lines) ← NEW
    ├── High-Content Analysis Widget
    ├── AI Segmentation Widget
    ├── Biophysics Widget
    ├── Interactive Plotting Widget
    └── Distance Tools Widget
```

---

## 📦 Current Widgets and Features

### 1. **File I/O Widget** (`file_io_widget.py`)
**Purpose:** Enhanced file loading beyond Napari's built-in capabilities

**Features:**
- Multi-format support (TIFF, PNG, JPEG, BMP, HDF5, MRC)
- Batch file loading
- Metadata preservation
- Series handling for multi-file sequences
- Integration with `aicsimageio` for advanced formats

**How it extends Napari:** Adds specialized microscopy format support

---

### 2. **Processing Widget** (`processing_widget.py` - 842 lines)
**Purpose:** Advanced image filtering and enhancement

**Features:**
- **Filtering:**
  - Gaussian blur (sigma adjustable)
  - Median filter (kernel size adjustable)
  - Bilateral filter (edge-preserving)
  - Morphological operations (erosion, dilation, opening, closing)
  
- **Enhancement:**
  - Contrast adjustment (CLAHE, histogram equalization)
  - Sharpening filters
  - Background subtraction
  - Intensity normalization

- **Deconvolution:**
  - Richardson-Lucy algorithm (threaded for large data)
  - Wiener filter
  - PSF generation

- **Advanced Features:**
  - AI-based denoising (when available)
  - Euclidean distance map generation (core/workflow backend)
  - Real-time preview
  - Batch processing mode

**How it extends Napari:** Adds comprehensive image processing pipeline with threaded operations

---

### 3. **Segmentation Widget** (`segmentation_widget.py` - 745 lines)
**Purpose:** Object detection and segmentation (Imaris Surfaces/Spots equivalent)

**Features:**
- **Spot Detection:**
  - Laplacian of Gaussian (LoG)
  - Difference of Gaussian (DoG)
  - Determinant of Hessian (DoH)
  - Size filtering (min/max diameter)
  - Quality thresholding
  - Results as Napari Points layer

- **Surface Creation:**
  - Marching Cubes algorithm for 3D surfaces
  - Smoothing factor control
  - Threshold-based segmentation
  - Results as Napari Surface layer

- **Watershed Segmentation:**
  - Automatic seed detection
  - Distance transform-based
  - Compactness parameter
  - Results as Napari Labels layer

- **Object Analysis:**
  - 30+ measurements per object (area, volume, intensity, shape)
  - Colocalization analysis
  - Distance calculations
  - CSV/JSON export

**How it extends Napari:** Adds quantitative spot detection and surface extraction with measurements

---

### 4. **Filament Tracing Widget** (`filament_tracing_widget.py` - 514 lines) ⭐ NEW
**Purpose:** Neuron/cytoskeleton tracing (Imaris FilamentTracer equivalent)

**Features:**
- **Tracing Methods:**
  - Skeletonization (2D/3D)
  - Ridge detection
  - Vesselness filtering
  - Centerline extraction

- **Parameters:**
  - Gaussian smoothing (sigma)
  - Threshold methods (Otsu, manual, adaptive)
  - Minimum filament length
  - Branch pruning

- **Analysis:**
  - Branch point detection
  - Segment length measurements
  - Filament thickness estimation
  - Network topology analysis
  - Total length, number of branches, endpoints

- **Output:**
  - Skeleton as Napari Shapes layer
  - Branch points as Napari Points layer
  - Statistics table
  - CSV/JSON export
  - Visualization overlays

- **Threading:** All operations threaded for large datasets

**How it extends Napari:** Adds complete neurite/filament tracing capabilities with quantification

---

### 5. **Cell Tracking & Lineage Widget** (`tracking_widget.py` - 684 lines) ⭐ NEW
**Purpose:** Multi-object tracking and lineage analysis (Imaris Track equivalent)

**Features:**
- **Tracking Algorithm:**
  - Hungarian algorithm for frame-to-frame linking
  - Cost matrix based on distance and size
  - Maximum linking distance parameter
  - Gap closing (up to N frames)
  - Track splitting/merging detection

- **Lineage Analysis:**
  - Cell division detection
  - Hierarchical lineage tree visualization
  - Parent-daughter relationships
  - Generation tracking
  - Tree export

- **Track Editing:**
  - Manual track correction
  - Split/merge tracks
  - Delete spurious tracks
  - Filter by track length
  - Track visualization with colors

- **Statistics:**
  - Track length (frames, distance)
  - Velocity (mean, max, instantaneous)
  - Displacement
  - Directionality
  - Division frequency
  - CSV/JSON export

- **Visualization:**
  - Tracks as Napari Tracks layer
  - Lineage tree plot (matplotlib)
  - Color-coded by generation/ID
  - Interactive selection

- **Threading:** All tracking operations run in background

**How it extends Napari:** Adds complete cell tracking with lineage analysis and quantification

---

### 6. **3D Visualization Widget** (`visualization_widget.py` - 153 lines)
**Purpose:** Enhanced 3D rendering controls (Imaris Volume Rendering equivalent)

**Features:**
- **Rendering Modes:**
  - Maximum Intensity Projection (MIP)
  - Translucent volume rendering
  - Isosurface rendering
  - Additive blending

- **Blending Modes:**
  - Translucent
  - Additive
  - Opaque

- **Adjustments:**
  - Colormap selection
  - Gamma correction (0.1 - 5.0)
  - Contrast limits (min/max)
  - Opacity control

- **Layer Management:**
  - Per-layer controls
  - Real-time updates
  - Multiple layers simultaneously

**How it extends Napari:** Provides unified controls for native Napari 3D rendering capabilities

---

### 7. **Analysis Widget** (`analysis_widget.py`)
**Purpose:** Colocalization and intensity analysis

**Features:**
- **Colocalization:**
  - Pearson correlation coefficient
  - Manders coefficients (M1, M2)
  - Overlap coefficient
  - Scatter plots
  - Thresholding options

- **Intensity Analysis:**
  - Profile plots
  - Histogram analysis
  - Region-based measurements
  - Time series extraction

- **Plotting:**
  - Interactive matplotlib plots
  - Export to PNG/SVG
  - Data export to CSV

**How it extends Napari:** Adds quantitative analysis beyond visualization

---

### 8. **Deconvolution Widget** (`deconvolution_widget.py`)
**Purpose:** Point Spread Function (PSF) deconvolution

**Features:**
- Richardson-Lucy algorithm
- Wiener deconvolution
- PSF estimation/loading
- Iteration control
- Progress tracking

**How it extends Napari:** Adds specialized microscopy image restoration

---

### 9. **Statistics Widget** (`statistics_widget.py`)
**Purpose:** Comprehensive statistical analysis (Imaris Statistics module equivalent)

**Features:**
- Object measurements aggregation
- Multi-channel statistics
- Time series analysis
- Statistical plots (histograms, box plots)
- Batch export
- Summary tables

**How it extends Napari:** Provides publication-ready statistical analysis

---

### 10. **High-Content Analysis Widget** (`hca_widget.py`)
**Purpose:** Multi-well plate analysis

**Features:**
- Well plate layout management
- Batch processing across wells
- Z-score normalization
- Quality control metrics
- Plate heatmaps
- Hit identification

**How it extends Napari:** Adds high-throughput screening capabilities

---

### 11. **AI Segmentation Widget** (`ai_segmentation_widget.py`)
**Purpose:** Deep learning-based segmentation

**Features:**
- Cellpose integration (when installed)
- StarDist integration (when installed)
- Model selection (cytoplasm, nuclei, custom)
- GPU acceleration
- Probability threshold adjustment
- Results as Napari Labels layer

**How it extends Napari:** Integrates state-of-the-art deep learning segmentation

---

### 12. **Interactive Plotting Widget** (`interactive_plotting_widget.py`)
**Purpose:** Linked scatter plots with viewer

**Features:**
- Scatter plots from measurements
- Click-to-highlight in viewer
- Selection synchronization
- Color coding by properties
- Export plots and data

**How it extends Napari:** Adds interactive data exploration

---

### 13. **Biophysics Widget** (`biophysics_widget.py`)
**Purpose:** Specialized biophysical measurements

**Features:**
- FRAP analysis (Fluorescence Recovery After Photobleaching)
- Photobleaching correction
- Diffusion coefficient calculation
- Time-lapse intensity tracking
- Exponential curve fitting

**How it extends Napari:** Adds specialized biophysical analysis tools

---

### 14. **MagicGUI Widgets** (`magicgui_analysis_widget.py`)
**Purpose:** Simple interactive widgets using magicgui

**Features:**
- Simple thresholding widget
- Adaptive thresholding widget
- Auto-generated UI from function signatures
- Direct integration with Napari layers

**How it extends Napari:** Demonstrates extensibility through magicgui

---

## 🎨 How PyMaris Extends Napari

### Native Napari Features Used:
- **Viewer:** Base window and 3D rendering engine
- **Layer System:** Image, Labels, Points, Shapes, Surface, Tracks
- **Dock Widgets:** All custom widgets use `viewer.window.add_dock_widget()`
- **Layer Properties:** Colormap, blending, contrast, visibility
- **Event System:** Layer insertion/removal callbacks
- **3D Visualization:** Built-in 3D rendering with controls

### Our Added Value:
1. **Quantitative Analysis:** 30+ object measurements, colocalization, tracking statistics
2. **Advanced Segmentation:** Spot detection, watershed, AI methods, filament tracing
3. **Tracking & Lineage:** Hungarian algorithm, gap closing, lineage trees
4. **Specialized Tools:** Deconvolution, FRAP analysis, HCA
5. **Export Capabilities:** CSV, JSON, TrackMate XML, publication plots
6. **Threaded Operations:** All heavy computations run in QThreads
7. **User-Friendly UI:** Organized dock widgets with progress indicators
8. **Imaris Compatibility:** Similar workflow and feature parity

---

## 🚀 Usage Example

```python
# Start PyMaris
python src/main_napari.py

# What happens:
1. Napari Viewer opens with custom title
2. 14 dock widgets added to left/right panels
3. Each widget provides specialized functionality
4. All results display as Napari layers
5. Full 3D navigation, layer controls from Napari
6. Quantitative data exportable from widgets
```

---

## 📊 Feature Comparison: PyMaris vs Imaris

| Feature | Imaris | PyMaris | Status |
|---------|--------|---------|--------|
| 3D Visualization | ✅ | ✅ | Complete |
| Volume Rendering (MIP) | ✅ | ✅ | Complete |
| Spot Detection | ✅ | ✅ | Complete |
| Surface Creation | ✅ | ✅ | Complete |
| Filament Tracing | ✅ | ✅ | Complete (514 lines) |
| Cell Tracking | ✅ | ✅ | Complete (684 lines) |
| Lineage Trees | ✅ | ✅ | Complete |
| Colocalization | ✅ | ✅ | Complete |
| Statistics | ✅ | ✅ | Complete |
| AI Denoising | ✅ | ✅ | Complete (ai_denoise backend with NLM/bilateral/Wiener) |
| Euclidean Distance Map Generation | ✅ | ✅ | Complete (restoration distance_map operation) |
| Distance Measurements | ✅ | ✅ | Complete (core backend metrics) |
| Distance Tools Widget | ✅ | ✅ | Complete (distance map + pairwise queries UI) |
| .ims File Support | ✅ | ✅ | Complete (AICS + h5py fallback) |
| Scene Management | ✅ | ✅ | Complete (scene listing/selection) |
| TrackMate XML Export | ✅ | ✅ | Complete (tracking export supports CSV + XML) |
| Python API | Limited | ✅ | Superior |
| Open Source | ❌ | ✅ | Advantage |
| Cost | $$$$ | Free | Advantage |

---

## 🔧 Technical Implementation

### Threading Architecture:
All time-consuming operations use QThread:
- `FilamentTracingThread` - skeleton extraction
- `TrackingThread` - Hungarian algorithm linking
- `SegmentationThread` - blob detection, watershed
- `DeconvolutionThread` - Richardson-Lucy iterations

### Signal-Slot Communication:
```python
# Example from tracking widget
self.thread.progress.connect(self.progress_bar.setValue)
self.thread.finished.connect(self.on_tracking_complete)
self.thread.error.connect(self.on_tracking_error)
```

### Napari Integration:
```python
# All results added as Napari layers
viewer.add_tracks(track_data, name="Cell Tracks")
viewer.add_points(spots, name="Detected Spots")
viewer.add_shapes(skeleton, name="Filament Skeleton")
viewer.add_labels(segmentation, name="Cells")
viewer.add_surface(surface_data, name="Cell Surface")
```

---

## 📁 File Structure

```
src/
├── main_napari.py              # Entry point - creates viewer + adds widgets
├── widgets/
│   ├── file_io_widget.py       # File loading
│   ├── processing_widget.py    # 842 lines - filters, enhancement
│   ├── segmentation_widget.py  # 745 lines - spots, surfaces, watershed
│   ├── filament_tracing_widget.py  # 514 lines - NEW
│   ├── tracking_widget.py      # 684 lines - NEW
│   ├── visualization_widget.py # 153 lines - 3D rendering controls
│   ├── analysis_widget.py      # Colocalization
│   ├── deconvolution_widget.py # PSF deconvolution
│   ├── statistics_widget.py    # Statistical analysis
│   ├── hca_widget.py          # High-content analysis
│   ├── ai_segmentation_widget.py  # Cellpose/StarDist
│   ├── interactive_plotting_widget.py  # Linked plots
│   ├── biophysics_widget.py    # FRAP analysis
│   └── magicgui_analysis_widget.py  # Simple widgets
└── utils/
    ├── image_utils.py          # Image validation, conversion
    └── analysis_utils.py       # Measurement calculations
```

---

## 🎓 Key Achievements

### ✅ Completed (Building on Napari):
1. **Volume Rendering** - MIP, alpha blending via Napari's native rendering
2. **Filament Tracing** - 514 lines of skeleton extraction, branch detection
3. **Cell Tracking** - 684 lines of Hungarian algorithm, lineage trees, gap closing
4. **Spot Detection** - LoG/DoG/DoH with size filtering
5. **Surface Creation** - Marching Cubes integration
6. **Segmentation** - Watershed, AI methods (Cellpose/StarDist)
7. **Colocalization** - Pearson, Manders coefficients
8. **Statistics** - 30+ measurements, export capabilities
9. **Threading** - All heavy operations non-blocking
10. **Documentation** - Comprehensive guides and API docs

### 🔭 Post-v1 Enhancements (Completed):
1. **Distance Transform Widget** - dedicated UI for custom pairwise queries
2. **Enhanced Export** - TrackMate XML export support added
3. **Scene Management UI** - scene picker integrated into file loading flow
4. **Advanced Plotting** - scatter/hexbin/histogram/box plot support

---

## 💡 Philosophy

**PyMaris is NOT a fork of Napari** - it's a **plugin/extension** that:
- Uses Napari as the visualization foundation
- Adds specialized scientific analysis widgets
- Integrates seamlessly with Napari's layer system
- Maintains compatibility with standard Napari plugins
- Provides Imaris-like workflow on an open-source platform

**Think of it as:** Napari (viewer) + PyMaris (analysis plugins) = Complete Image Analysis Suite

---

## 🔍 Next Steps

To see all functionality in action:
1. Run `start.bat` (Windows) or `python src/main_napari.py`
2. Load a 3D/4D image via File I/O widget
3. Explore each dock widget on the left and right panels
4. All 14 widgets are active and functional
5. Results appear as Napari layers in the layer list

**The functionality is all there** - it's just organized as modular dock widgets rather than a single monolithic application!
