# PyMaris Scientific Image Analyzer - Complete Feature Set

## Overview
PyMaris is a comprehensive multidimensional image analysis program designed to replicate and extend the functionality of Bitplane Imaris. It provides advanced tools for fluorescence and electron microscopy image analysis with a focus on usability and scientific rigor.

## Version Information
- **Version**: 2.1.0
- **Last Updated**: October 2025
- **License**: MIT

## Key Features Replicating Imaris

### 1. Volume Rendering & Visualization
The Volume Rendering widget provides Imaris-like 3D visualization capabilities:

#### Maximum Intensity Projection (MIP)
- Project 3D volumes along X, Y, or Z axes
- Adjustable projection range (slice selection)
- Real-time contrast and brightness adjustment
- Support for partial volume MIP

#### Alpha Blending Volume Rendering
- Multiple rendering modes:
  - Composite (standard volume rendering)
  - Average intensity projection
  - Maximum intensity projection
  - Minimum intensity projection
  - Attenuated MIP (depth-dependent)
- Adjustable global opacity
- Opacity threshold control
- Variable sampling rate for quality/speed tradeoff

#### Orthogonal Slice Views
- Simultaneous XY, XZ, and YZ cross-sections
- Interactive position control
- Real-time slice updates
- Automatic centering option

#### Volume Clipping
- Independent clipping planes for X, Y, Z axes
- Percentage-based clipping controls
- Real-time preview of clipped volumes
- Ideal for revealing internal structures

**Usage Example:**
```python
# Load a 3D volume
viewer = napari.Viewer()
volume = np.random.random((50, 512, 512))
viewer.add_image(volume, name="My Volume")

# Open Volume Rendering widget
# Select "My Volume" from dropdown
# Choose "MIP Rendering" tab
# Click "Generate MIP" for Z-axis projection
```

### 2. Filament Tracing (Similar to FilamentTracer)
Advanced filament analysis for neurons, cytoskeleton, and fibrous structures:

#### Features
- Automated filament detection and tracing
- Skeleton extraction using morphological thinning
- Branch point detection and analysis
- Filament thickness measurement
- Path extraction and ordering
- Individual filament statistics

#### Analysis Metrics
- Total filament length
- Number of filaments
- Number of branch points
- Average filament thickness
- Individual filament lengths

#### Export Options
- Skeleton images (TIFF format)
- Statistical data (CSV format)
- Branch point coordinates

**Usage Example:**
```python
# Load a filament image (e.g., neurons, actin)
image = imread("neurons.tif")
viewer.add_image(image, name="Neurons")

# Open Filament Tracing widget
# Adjust Gaussian smoothing (typically 1.0)
# Select threshold method (Otsu recommended)
# Click "Trace Filaments"
# Export skeleton or statistics as needed
```

### 3. Advanced Cell Tracking & Lineage
Comprehensive particle/cell tracking with lineage analysis:

#### Tracking Capabilities
- Frame-to-frame linking using Hungarian algorithm
- Gap closing for interrupted tracks
- Minimum track length filtering
- Maximum search distance control

#### Lineage Analysis
- Automatic cell division detection
- Cell merge detection
- Hierarchical lineage tree visualization
- Parent-daughter relationship tracking

#### Track Statistics
- Total number of tracks
- Mean track length
- Mean displacement
- Mean velocity
- Track straightness (displacement/path length)
- Individual track properties

#### Export Formats
- Track coordinates (CSV)
- Lineage information (JSON)
- napari-compatible track layers

**Usage Example:**
```python
# Load segmented time series
labels_4d = np.zeros((20, 512, 512), dtype=int)  # 20 timepoints
# ... fill with segmented objects ...
viewer.add_labels(labels_4d, name="Cells")

# Open Cell Tracking & Lineage widget
# Set max distance (e.g., 50 pixels)
# Set max gap frames (e.g., 2)
# Enable "Detect cell divisions"
# Click "Track Objects"
# View lineage tree in right panel
# Export tracks or lineage data
```

### 4. Spot Detection & Analysis
Enhanced spot detection with comprehensive statistics:

#### Methods
- LoG (Laplacian of Gaussian)
- DoG (Difference of Gaussian)
- DoH (Determinant of Hessian)

#### Features
- Multi-scale blob detection
- Adjustable sigma range
- Threshold control
- Automatic spot counting

#### Statistics
- Spot count
- Average spot intensity
- Spot size distribution
- Spatial distribution analysis

### 5. Surface Rendering
3D surface extraction and analysis:

#### Capabilities
- Marching cubes algorithm
- Adjustable isosurface level
- Automatic level calculation
- Surface area measurement
- Vertex and face count

### 6. Colocalization Analysis
Statistical colocalization between channels:

#### Coefficients
- Pearson correlation
- Spearman correlation
- Manders coefficients (M1, M2)
- Overlap coefficients
- Costes automatic thresholding

#### Visualizations
- Scatter plots
- Intensity histograms
- Colocalization masks
- Statistical reports

### 7. Image Processing Suite
Comprehensive image enhancement and filtering:

#### Smoothing & Filtering
- Gaussian filter (adjustable sigma)
- Median filter (noise reduction)
- Bilateral filter (edge-preserving)

#### Thresholding
- Manual threshold
- Automatic methods (Otsu, Li, Yen, Triangle, Mean, Minimum)
- Adaptive/local thresholding
- Block size and offset control

#### Contrast Enhancement
- Histogram equalization
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gamma adjustment
- Adaptive gamma

#### Morphological Operations
- Erosion
- Dilation
- Opening (remove small objects)
- Closing (fill holes)
- Top-hat transform
- Adjustable structuring element size

#### Deconvolution
- Richardson-Lucy algorithm
- Wiener filter
- PSF generation (Gaussian, custom)
- Iteration control

### 8. Statistical Analysis
Comprehensive statistics module:

#### Object Measurements
- Area
- Perimeter
- Circularity
- Eccentricity
- Major/minor axis lengths
- Orientation
- Solidity
- Extent
- Feret diameter

#### Intensity Measurements
- Mean intensity
- Max/min intensity
- Standard deviation
- Integrated intensity
- Weighted centroid

#### Export
- CSV format
- All measurements in tabular form
- Compatible with Excel, R, Python

### 9. Multi-Format File I/O
Support for major microscopy file formats:

#### Import Formats
- TIFF/TIF (8-bit, 16-bit, multi-page)
- CZI (Carl Zeiss)
- LIF (Leica)
- ND2 (Nikon)
- OIB/OIF (Olympus)
- LSM (Zeiss confocal)
- IMS (Imaris)

#### Export Formats
- TIFF (standard, ImageJ-compatible)
- MRC (for ChimeraX)
- CSV (measurements, coordinates)
- JSON (metadata)
- TrackMate XML (for Fiji)
- HDF5

### 10. Batch Processing
Efficient multi-file analysis:

#### Features
- Batch file upload
- Workflow templates
- Progress tracking
- Parallel processing
- Result aggregation

## Installation

### Requirements
```bash
# Core dependencies
pip install numpy scipy scikit-image matplotlib pandas

# Napari desktop interface
pip install napari[all] PyQt6

# Optional AI/ML features
pip install cellpose stardist

# Optional file format support
pip install aicsimageio tifffile mrcfile

# Optional tracking
pip install btrack
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/mhendzel2/3DIA.git
cd 3DIA

# Install dependencies
pip install -r requirements.txt

# Launch Napari interface
python src/main_napari.py

# Or launch web interface
python src/main.py enhanced
```

## Usage Workflows

### Workflow 1: 3D Cell Segmentation and Tracking
1. Load 4D image (TZYX)
2. Use Segmentation widget → Apply Cellpose or watershed
3. Use Cell Tracking widget → Configure parameters and track
4. View lineage tree
5. Export tracks to CSV

### Workflow 2: Neuron Tracing
1. Load neuron image (2D or 3D)
2. Use Image Processing → Enhance contrast
3. Use Filament Tracing → Trace with Otsu threshold
4. Analyze branch points
5. Export skeleton and statistics

### Workflow 3: Volume Rendering
1. Load 3D confocal stack
2. Use Volume Rendering → MIP tab
3. Generate XY MIP
4. Adjust contrast for publication
5. Save as image

### Workflow 4: Colocalization Analysis
1. Load multi-channel image
2. Split channels if needed
3. Use Analysis widget → Colocalization tab
4. Select two channels
5. View scatter plot and coefficients
6. Export statistics

### Workflow 5: FIB-SEM Analysis
1. Load FIB-SEM volume
2. Use Volume Rendering → Clipping to explore
3. Use Segmentation → Surface creation
4. Measure surface area and volume
5. Export to MRC for ChimeraX

## Advanced Features

### High-Content Analysis (HCA)
- Multi-well plate analysis
- Dose-response curves
- Z-score calculation
- Hit identification

### FIB-SEM Specialized Tools
- 3D object counting
- Tomographic slice analysis
- Membrane segmentation
- GPU-accelerated processing

### Biophysics Module
- FRAP recovery analysis
- Photobleaching correction
- Time series analysis

## Comparison with Imaris

| Feature | PyMaris | Imaris |
|---------|---------|--------|
| Volume Rendering | ✓ | ✓ |
| MIP | ✓ | ✓ |
| Orthogonal Views | ✓ | ✓ |
| Spot Detection | ✓ | ✓ |
| Surface Creation | ✓ | ✓ |
| Cell Tracking | ✓ | ✓ |
| Lineage Analysis | ✓ | ✓ |
| Filament Tracing | ✓ | ✓ |
| Colocalization | ✓ | ✓ |
| Statistics | ✓ | ✓ |
| Batch Processing | ✓ | ✓ |
| Python API | ✓ | Limited |
| Open Source | ✓ | ✗ |
| Cost | Free | $$$$ |

## Troubleshooting

### Common Issues

**Issue: Napari won't start**
```bash
# Try updating Qt
pip install --upgrade PyQt6

# Or use PyQt5
pip uninstall PyQt6
pip install PyQt5
```

**Issue: Out of memory errors**
```bash
# Process data in chunks
# Or use Volume Clipping to reduce data size
```

**Issue: Slow tracking**
```bash
# Reduce max_distance parameter
# Disable gap closing
# Filter short tracks after analysis
```

## API Documentation

### Programmatic Usage

```python
import napari
from widgets.volume_rendering_widget import VolumeRenderingWidget

# Create viewer
viewer = napari.Viewer()

# Add data
volume = load_data("mydata.tif")
viewer.add_image(volume)

# Add widget programmatically
vr_widget = VolumeRenderingWidget(viewer)
viewer.window.add_dock_widget(vr_widget, name="Volume Rendering")

# Run napari
napari.run()
```

### Scripting Analysis

```python
from src.advanced_analysis import AdvancedSegmentation
from src.timelapse_processor import ImageAligner

# Segment with advanced methods
segmenter = AdvancedSegmentation()
labels = segmenter.morphological_snakes(image, iterations=100)

# Align time series
aligner = ImageAligner()
result = aligner.align_timelapse_sequence(
    image_sequence=frames,
    method='phase_correlation'
)
```

## Contributing
Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Citation
If you use PyMaris in your research, please cite:

```
@software{pymaris2025,
  title = {PyMaris: Open-Source Multidimensional Image Analysis},
  author = {Henderson, Michael},
  year = {2025},
  url = {https://github.com/mhendzel2/3DIA}
}
```

## Support
- GitHub Issues: https://github.com/mhendzel2/3DIA/issues
- Documentation: See COMPLETE_PACKAGE_README.md
- Installation Guide: See NAPARI_INSTALLATION_GUIDE.md

## License
MIT License - see LICENSE file for details

## Acknowledgments
- Inspired by Bitplane Imaris
- Built on the Napari platform
- Uses scikit-image, scipy, and other open-source tools
- Community contributions welcome
