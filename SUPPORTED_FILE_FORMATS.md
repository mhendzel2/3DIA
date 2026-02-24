# PyMaris Supported File Formats

## Installation Model

PyMaris uses one installer:

```batch
install.bat
```

That installer includes core format support, including:
- Imaris `.ims` via `h5py`
- Zarr via `zarr`
- Zeiss CZI via `czifile`
- Leica LIF via `readlif`
- Nikon ND2 via `nd2reader`

## Supported Formats

### Microscopy / Scientific

| Format | Extension | Primary Reader |
|--------|-----------|----------------|
| TIFF / OME-TIFF | `.tif`, `.tiff`, `.ome.tif` | `tifffile` |
| Zeiss CZI | `.czi` | `czifile` |
| Leica LIF | `.lif` | `readlif` |
| Nikon ND2 | `.nd2` | `nd2reader` |
| Imaris | `.ims` | `h5py` |
| MRC/CCP4 | `.mrc`, `.map` | `mrcfile` |
| Zarr | `.zarr` | `zarr` |
| HDF5 | `.h5`, `.hdf5` | `h5py` |

### Common Image / Video

| Format | Extension | Reader |
|--------|-----------|--------|
| PNG | `.png` | `imageio` / `Pillow` |
| JPEG | `.jpg`, `.jpeg` | `imageio` / `Pillow` |
| BMP | `.bmp` | `imageio` / `Pillow` |
| GIF | `.gif` | `imageio` |
| AVI / MP4 / MOV | `.avi`, `.mp4`, `.mov` | `opencv-python` |

## Reader Preference

When multiple readers are available, PyMaris prefers specialized readers first, then falls back to general readers:
1. Format-specific readers (`czifile`, `readlif`, `nd2reader`)
2. `tifffile` / `mrcfile` / `h5py` / `zarr`
3. `pims`
4. `imageio` / `Pillow`

## Verification

```batch
venv\Scripts\python.exe -c "import tifffile,czifile,readlif,nd2reader,mrcfile,h5py,zarr,pims,imageio; print('Format stack OK')"
```

## If a Format Fails to Open

1. Re-run:
```batch
install.bat
```
2. Run:
```batch
test-installation.bat
```
3. Convert source data to TIFF/OME-TIFF as a fallback for interoperability.
