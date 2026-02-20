# PyMaris Supported File Formats

## Overview

PyMaris supports a wide range of microscopy and image file formats through multiple reader libraries. The system automatically selects the best reader for each file type.

## Installation

### Basic Installation (TIFF, PNG, JPEG, BMP)
```batch
install_minimal.bat
```

### Full Format Support
```batch
install_formats.bat
```

This installs all format-specific readers that don't require Java or C++ compilation.

## Supported Formats

### ✅ Microscopy Formats (Proprietary)

| Format | Extension | Library | Installation | Notes |
|--------|-----------|---------|--------------|-------|
| **Zeiss CZI** | `.czi` | czifile | `pip install czifile` | Pure Python, very reliable |
| **Leica LIF** | `.lif` | readlif | `pip install readlif` | Pure Python, works well |
| **Nikon ND2** | `.nd2` | nd2reader | `pip install nd2reader` | May have dependencies |
| **Imaris** | `.ims` | h5py | `pip install h5py` | HDF5-based format |
| **Olympus OIB/OIF** | `.oib` `.oif` | aicsimageio or convert | Use ImageJ to convert | |

### ✅ Standard Microscopy Formats

| Format | Extension | Library | Installation | Notes |
|--------|-----------|---------|--------------|-------|
| **TIFF** | `.tif` `.tiff` | tifffile | `pip install tifffile` | Best for general use |
| **OME-TIFF** | `.ome.tif` | tifffile | `pip install tifffile` | Preserves metadata |
| **ImageJ TIFF** | `.tif` | tifffile | `pip install tifffile` | With ImageJ metadata |
| **LSM** | `.lsm` | tifffile | `pip install tifffile` | Zeiss LSM format |
| **STK** | `.stk` | tifffile | `pip install tifffile` | MetaMorph stacks |

### ✅ Electron Microscopy

| Format | Extension | Library | Installation | Notes |
|--------|-----------|---------|--------------|-------|
| **MRC** | `.mrc` `.map` | mrcfile | `pip install mrcfile` | Standard EM format |
| **DM3/DM4** | `.dm3` `.dm4` | hyperspy or convert | Complex | Use ImageJ to convert |

### ✅ Common Image Formats

| Format | Extension | Library | Installation | Notes |
|--------|-----------|---------|--------------|-------|
| **PNG** | `.png` | imageio/pillow | Always installed | Lossless compression |
| **JPEG** | `.jpg` `.jpeg` | imageio/pillow | Always installed | Lossy compression |
| **BMP** | `.bmp` | imageio/pillow | Always installed | Uncompressed |
| **GIF** | `.gif` | imageio | `pip install imageio` | Animations supported |

### ✅ Large Array Formats

| Format | Extension | Library | Installation | Notes |
|--------|-----------|---------|--------------|-------|
| **Zarr** | `.zarr` (dir) | zarr | `pip install zarr` | Chunked, compressed arrays |
| **HDF5** | `.h5` `.hdf5` | h5py | `pip install h5py` | Hierarchical data format |
| **NumPy** | `.npy` `.npz` | numpy | Always installed | Python arrays |

### ✅ Video Formats (via OpenCV)

| Format | Extension | Library | Installation | Notes |
|--------|-----------|---------|--------------|-------|
| **AVI** | `.avi` | opencv-python | `pip install opencv-python` | Video files |
| **MP4** | `.mp4` | opencv-python | `pip install opencv-python` | Compressed video |
| **MOV** | `.mov` | opencv-python | `pip install opencv-python` | QuickTime |

## Reader Priority

PyMaris uses this priority order when multiple readers can handle a format:

1. **Format-specific readers** (czifile, readlif, nd2reader, etc.) - Most reliable
2. **aicsimageio** (if installed) - Universal but requires compilation
3. **tifffile** - For TIFF files
4. **mrcfile** - For MRC files
5. **h5py** - For HDF5/Imaris files
6. **zarr** - For Zarr arrays
7. **pims** - Universal fallback
8. **imageio** - Common formats
9. **PIL/Pillow** - Basic formats

## Installation Commands

### Minimal Installation
```batch
# Core functionality only
install_minimal.bat

# Result: TIFF, PNG, JPEG, BMP, MRC support
```

### Format-Specific Readers
```batch
# All pure-Python readers (recommended)
install_formats.bat

# Result: + CZI, LIF, ND2, Imaris, Zarr support
```

### Individual Format Installation
```bash
# Activate venv first
venv\Scripts\activate.bat

# Zeiss CZI files
pip install czifile

# Leica LIF files
pip install readlif

# Nikon ND2 files
pip install nd2reader

# Imaris .ims files
pip install h5py

# Zarr arrays
pip install zarr

# Universal format support
pip install pims imageio

# Additional formats
pip install opencv-python
```

## Format Recommendations

### For Daily Use
**Use:** TIFF or OME-TIFF
- **Why:** Universal support, preserves metadata, no compression artifacts
- **Tool:** Save from ImageJ/FIJI or your microscope software

### For Large Datasets
**Use:** Zarr or HDF5/Imaris
- **Why:** Chunked access, compression, multi-resolution
- **Tool:** Convert using Python scripts or PyMaris

### For Sharing
**Use:** OME-TIFF or PNG
- **Why:** Widely supported, preserves information
- **Tool:** Export from PyMaris

### For Publication Figures
**Use:** TIFF or PNG
- **Why:** No compression artifacts, accepted by journals
- **Tool:** Export from PyMaris with appropriate contrast

## Troubleshooting

### "No suitable reader found"

**Solution 1:** Install format-specific reader
```batch
venv\Scripts\activate.bat
pip install czifile  # For CZI files
pip install readlif  # For LIF files
pip install h5py     # For Imaris files
```

**Solution 2:** Convert to TIFF using ImageJ/FIJI
1. Open ImageJ/FIJI
2. Open your proprietary file (Bio-Formats imports automatically)
3. File → Save As → TIFF
4. Open in PyMaris

### "Failed to load with any available reader"

Check which readers are installed:
```python
# In PyMaris, check available readers widget
# Or run in Python:
import sys
readers = []

for module in ['tifffile', 'czifile', 'readlif', 'nd2reader', 
               'mrcfile', 'h5py', 'zarr', 'imageio', 'PIL']:
    try:
        __import__(module)
        readers.append(module)
    except ImportError:
        pass

print(f"Available: {', '.join(readers)}")
```

### Imaris Files Won't Open

Imaris files are HDF5 containers with complex internal structure.

**Option 1:** Install h5py
```batch
pip install h5py
```

**Option 2:** Export from Imaris
- File → Export → TIFF Series
- Then open in PyMaris

**Option 3:** Use aicsimageio (requires C++ build tools)
```batch
# If you have C++ build tools installed
pip install aicsimageio
```

### Zarr Files Not Loading

Zarr files can be directories or zip files.

**Check:**
1. Path points to `.zarr` directory
2. Directory contains valid Zarr structure
3. `pip install zarr` was successful

**Example:**
```python
import zarr
z = zarr.open('mydata.zarr', 'r')
print(z.tree())  # Show structure
```

## Converting Between Formats

### TIFF → Zarr (for large files)
```python
import tifffile
import zarr
import numpy as np

# Read TIFF
data = tifffile.imread('large_file.tif')

# Save as Zarr with compression
z = zarr.open('output.zarr', mode='w', 
              shape=data.shape, dtype=data.dtype,
              chunks=(1, 256, 256),
              compressor=zarr.Blosc(cname='zstd', clevel=3))
z[:] = data
```

### Proprietary → TIFF (via ImageJ macro)
```javascript
// ImageJ macro for batch conversion
input = "C:/input/folder/";
output = "C:/output/folder/";

list = getFileList(input);
for (i = 0; i < list.length; i++) {
    open(input + list[i]);
    saveAs("Tiff", output + replace(list[i], ".czi", ".tif"));
    close();
}
```

### Imaris → OME-TIFF
```python
import h5py
import tifffile

# Read from Imaris
with h5py.File('data.ims', 'r') as f:
    data = f['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'][:]

# Save as OME-TIFF
tifffile.imwrite('data.ome.tif', data, 
                 metadata={'axes': 'ZYX'}, 
                 photometric='minisblack')
```

## Format Specifications

### Imaris .ims Structure
```
file.ims (HDF5)
├── DataSet
│   ├── ResolutionLevel 0
│   │   ├── TimePoint 0
│   │   │   ├── Channel 0
│   │   │   │   └── Data (array)
│   │   │   ├── Channel 1
│   │   │   │   └── Data (array)
│   ├── ResolutionLevel 1 (lower resolution)
├── DataSetInfo
│   ├── Image (attributes with voxel sizes)
│   ├── TimeInfo
```

### Zarr Structure
```
data.zarr/
├── .zarray (JSON metadata)
├── .zattrs (JSON attributes)
├── 0.0.0 (chunk files)
├── 0.0.1
├── ...
```

### OME-TIFF Metadata
```xml
<OME>
  <Image>
    <Pixels DimensionOrder="XYZCT"
            SizeX="1024" SizeY="1024" SizeZ="50"
            SizeC="3" SizeT="10"
            PhysicalSizeX="0.1" PhysicalSizeY="0.1" PhysicalSizeZ="0.3"
            PhysicalSizeXUnit="µm" ...>
    </Pixels>
  </Image>
</OME>
```

## Best Practices

### 1. Always use TIFF for archival
- Widely supported
- No vendor lock-in
- Preserves metadata (OME-TIFF)

### 2. Use Zarr for very large datasets (>10GB)
- Chunked access (don't load entire file)
- Compression saves disk space
- Multi-resolution pyramids

### 3. Keep original proprietary files
- Highest quality
- Full metadata
- Future compatibility

### 4. Document your conversion
- Record software versions
- Note any processing applied
- Keep conversion scripts

## Quick Reference

| What You Have | What to Install | Command |
|---------------|----------------|---------|
| Zeiss microscope | CZI reader | `pip install czifile` |
| Leica microscope | LIF reader | `pip install readlif` |
| Nikon microscope | ND2 reader | `pip install nd2reader` |
| Imaris user | HDF5 reader | `pip install h5py` |
| Large datasets | Zarr support | `pip install zarr` |
| Everything | Full support | `install_formats.bat` |

## Getting Help

If a format isn't working:

1. **Check installation:**
   ```batch
   venv\Scripts\pip list | findstr "image"
   ```

2. **Try alternative:**
   - Convert to TIFF in ImageJ/FIJI
   - Use format-specific software to export

3. **Report issue:**
   - Include file extension
   - Note error message
   - List installed readers

**The enhanced file I/O widget will show you which readers are available and automatically use the best one for each file!**
