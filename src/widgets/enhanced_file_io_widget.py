"""
Enhanced File I/O Widget - Works without aicsimageio or bioformats
Uses format-specific readers with intelligent fallbacks
"""

import os
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QComboBox, QSpinBox, 
                             QCheckBox, QGroupBox, QProgressBar, QTextEdit, QInputDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import napari
from napari.layers import Image

# Import available readers with fallbacks
available_readers = {}

# Standard libraries (should always be available)
try:
    import tifffile
    available_readers['tifffile'] = 'TIFF files (fast, reliable)'
except ImportError:
    pass

try:
    import imageio
    available_readers['imageio'] = 'Common formats (PNG, JPEG, etc.)'
except ImportError:
    pass

try:
    from PIL import Image as PILImage
    available_readers['pillow'] = 'Basic formats (PNG, JPEG, BMP)'
except ImportError:
    pass

try:
    import mrcfile
    available_readers['mrcfile'] = 'MRC files (electron microscopy)'
except ImportError:
    pass

# Format-specific readers (install as needed)
try:
    import czifile
    available_readers['czifile'] = 'Zeiss CZI files'
except ImportError:
    pass

try:
    import readlif
    available_readers['readlif'] = 'Leica LIF files'
except ImportError:
    pass

try:
    import nd2reader
    available_readers['nd2reader'] = 'Nikon ND2 files'
except ImportError:
    pass

try:
    import pims
    available_readers['pims'] = 'Many formats via pims'
except ImportError:
    pass

try:
    import zarr
    available_readers['zarr'] = 'Zarr arrays (large datasets)'
except ImportError:
    pass

try:
    import h5py
    available_readers['h5py'] = 'HDF5 and Imaris .ims files'
except ImportError:
    pass

# Optional advanced readers
try:
    from aicsimageio import AICSImage
    available_readers['aicsimageio'] = 'All microscopy formats'
except ImportError:
    pass

print(f"Available image readers: {', '.join(available_readers.keys())}")
if not available_readers:
    print("WARNING: No image reading libraries found!")
    print("Install at least: pip install tifffile imageio pillow")

class SmartFileLoader:
    """Smart file loader that uses the best available reader for each format"""
    
    @staticmethod
    def load_image(file_path, load_options=None):
        """Load image using the best available reader"""
        file_path = Path(file_path)
        load_options = load_options or {}
        scene = load_options.get('scene')
        scene_index = load_options.get('scene_index')
        ext = file_path.suffix.lower()
        
        # Try format-specific readers first (most reliable)
        if ext == '.czi' and 'czifile' in available_readers:
            return SmartFileLoader._load_with_czifile(file_path)
        
        if ext == '.lif' and 'readlif' in available_readers:
            return SmartFileLoader._load_with_readlif(file_path)
        
        if ext == '.nd2' and 'nd2reader' in available_readers:
            return SmartFileLoader._load_with_nd2reader(file_path)
        
        if ext in ['.mrc', '.map'] and 'mrcfile' in available_readers:
            return SmartFileLoader._load_with_mrcfile(file_path)
        
        if ext in ['.tif', '.tiff', '.lsm'] and 'tifffile' in available_readers:
            return SmartFileLoader._load_with_tifffile(file_path)
        
        if ext == '.zarr' and 'zarr' in available_readers:
            return SmartFileLoader._load_with_zarr(file_path)
        
        if ext == '.ims' and 'aicsimageio' in available_readers:
            return SmartFileLoader._load_with_aicsimageio(file_path, scene=scene, scene_index=scene_index)
        
        if ext == '.ims' and 'h5py' in available_readers:
            return SmartFileLoader._load_with_imaris(file_path, scene=scene, scene_index=scene_index)
        
        # Try aicsimageio for everything (if available)
        if 'aicsimageio' in available_readers:
            return SmartFileLoader._load_with_aicsimageio(file_path, scene=scene, scene_index=scene_index)
        
        # Try pims as universal fallback
        if 'pims' in available_readers:
            return SmartFileLoader._load_with_pims(file_path)
        
        # Try imageio for common formats
        if 'imageio' in available_readers:
            return SmartFileLoader._load_with_imageio(file_path)
        
        # Last resort: PIL/Pillow
        if 'pillow' in available_readers:
            return SmartFileLoader._load_with_pillow(file_path)
        
        raise Exception(f"No suitable reader found for {ext} files")
    
    @staticmethod
    def _load_with_czifile(file_path):
        """Load CZI file with czifile"""
        import czifile
        
        with czifile.CziFile(file_path) as czi:
            data = czi.asarray()
            
        # Remove singleton dimensions
        data = np.squeeze(data)
        
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': 'czifile',
            'scale': [1.0] * data.ndim
        }
        
        return data, metadata
    
    @staticmethod
    def _load_with_readlif(file_path):
        """Load LIF file with readlif"""
        from readlif.reader import LifFile
        
        lif = LifFile(str(file_path))
        
        # Get first image (you could extend this to handle multiple images)
        img_list = list(lif.get_iter_image())
        if not img_list:
            raise Exception("No images found in LIF file")
        
        image = img_list[0]
        data = np.array(image)
        
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': 'readlif',
            'scale': [1.0] * data.ndim
        }
        
        return data, metadata
    
    @staticmethod
    def _load_with_nd2reader(file_path):
        """Load ND2 file with nd2reader"""
        from nd2reader import ND2Reader
        
        with ND2Reader(str(file_path)) as images:
            # Get all data
            data = np.array(images)
            
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': 'nd2reader',
            'scale': [1.0] * data.ndim
        }
        
        return data, metadata
    
    @staticmethod
    def _load_with_mrcfile(file_path):
        """Load MRC file with mrcfile"""
        import mrcfile
        
        with mrcfile.open(file_path) as mrc:
            data = mrc.data.copy()
            
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': 'mrcfile',
            'scale': [1.0] * data.ndim
        }
        
        return data, metadata
    
    @staticmethod
    def _load_with_tifffile(file_path):
        """Load TIFF file with tifffile"""
        import tifffile
        
        reader_name = 'tifffile'
        try:
            data = tifffile.imread(file_path)
        except Exception as exc:
            message = str(exc).lower()
            if 'imagecodecs' in message and 'lzw' in message and 'pillow' in available_readers:
                from PIL import Image as PILImage

                with PILImage.open(file_path) as pil_image:
                    data = np.asarray(pil_image)
                reader_name = 'pillow (LZW fallback)'
            elif 'imagecodecs' in message and 'lzw' in message:
                raise RuntimeError(
                    f"Failed to load {file_path}: TIFF uses LZW compression. "
                    "Install `imagecodecs` to enable decoding."
                ) from exc
            else:
                raise
        
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': reader_name,
            'scale': [1.0] * data.ndim
        }
        
        # Try to get pixel sizes from ImageJ metadata
        try:
            with tifffile.TiffFile(file_path) as tif:
                if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                    ij_meta = tif.imagej_metadata
                    if 'spacing' in ij_meta:
                        metadata['scale'] = [ij_meta['spacing']] * data.ndim
        except:
            pass
            
        return data, metadata
    
    @staticmethod
    def _load_with_zarr(file_path):
        """Load Zarr array"""
        import zarr
        
        # Zarr can be a directory or a zip file
        z = zarr.open(str(file_path), mode='r')
        
        # Get the array (might be nested)
        if isinstance(z, zarr.hierarchy.Group):
            # Find first array in group
            array_keys = list(z.array_keys())
            if not array_keys:
                raise Exception("No arrays found in Zarr group")
            data = z[array_keys[0]][:]
        else:
            data = z[:]
        
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': 'zarr',
            'scale': [1.0] * data.ndim
        }
        
        # Try to get metadata from Zarr attributes
        try:
            if hasattr(z, 'attrs'):
                if 'voxel_size' in z.attrs:
                    metadata['scale'] = list(z.attrs['voxel_size'])
                elif 'pixelsize' in z.attrs:
                    metadata['scale'] = list(z.attrs['pixelsize'])
        except:
            pass
        
        return data, metadata
    
    @staticmethod
    def _load_with_imaris(file_path, scene=None, scene_index=None):
        """Load Imaris .ims file (HDF5 format)"""
        try:
            from pymaris.io import list_scenes as core_list_scenes
            from pymaris.io import open_image as core_open_image
        except Exception:
            core_list_scenes = None
            core_open_image = None
        
        if core_open_image is not None:
            kwargs = {}
            if scene is not None:
                kwargs['scene'] = str(scene)
            if scene_index is not None:
                kwargs['scene_index'] = int(scene_index)
            image = core_open_image(file_path, **kwargs)
            data = np.asarray(image.as_numpy())
            scale = list(image.scale_for_axes())
            metadata = {
                'name': file_path.stem,
                'file_path': str(file_path),
                'shape': data.shape,
                'dtype': data.dtype,
                'reader': 'pymaris.io (Imaris)',
                'scale': scale,
                'axes': list(image.axes),
                'scene': image.metadata.get('scene'),
                'available_scenes': core_list_scenes(file_path) if core_list_scenes is not None else [],
                'pixel_size': dict(image.pixel_size),
            }
            return data, metadata
        
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Imaris files have a specific structure
            # DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data
            
            # Find the data path
            data_path = None
            
            # Try common Imaris paths
            if 'DataSet' in f:
                dataset = f['DataSet']
                
                # Try to find resolution level 0
                if 'ResolutionLevel 0' in dataset:
                    res_level = dataset['ResolutionLevel 0']
                    
                    # Try to find TimePoint 0
                    if 'TimePoint 0' in res_level:
                        timepoint = res_level['TimePoint 0']
                        
                        # Try to find Channel 0
                        if 'Channel 0' in timepoint:
                            channel = timepoint['Channel 0']
                            
                            # Get the data
                            if 'Data' in channel:
                                data = channel['Data'][:]
                            else:
                                data = channel[:]
                        else:
                            # No channels, just get data
                            if 'Data' in timepoint:
                                data = timepoint['Data'][:]
                            else:
                                data = timepoint[:]
                    else:
                        # No timepoints, try direct access
                        keys = list(res_level.keys())
                        if keys:
                            data = res_level[keys[0]][:]
                else:
                    # Try alternative structure
                    keys = list(dataset.keys())
                    if keys:
                        data = dataset[keys[0]][:]
            else:
                # Fallback: get first dataset found
                def find_first_dataset(group):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            return item[:]
                        elif isinstance(item, h5py.Group):
                            result = find_first_dataset(item)
                            if result is not None:
                                return result
                    return None
                
                data = find_first_dataset(f)
                
                if data is None:
                    raise Exception("Could not find image data in Imaris file")
            
            metadata = {
                'name': file_path.stem,
                'file_path': str(file_path),
                'shape': data.shape,
                'dtype': data.dtype,
                'reader': 'h5py (Imaris)',
                'scale': [1.0] * data.ndim,
                'scene': scene,
            }
            
            # Try to extract voxel size from Imaris metadata
            try:
                if 'DataSetInfo' in f:
                    info = f['DataSetInfo']
                    if 'Image' in info.attrs:
                        image_info = info.attrs['Image']
                        # Parse voxel sizes if available
                        # This is file-format specific and may vary
            except:
                pass
        
        return data, metadata
    
    @staticmethod
    def _load_with_aicsimageio(file_path, scene=None, scene_index=None):
        """Load with aicsimageio while preserving microscopy dimensions."""
        from aicsimageio import AICSImage
        
        img = AICSImage(file_path)
        scenes = [str(value) for value in getattr(img, "scenes", [])]
        selected_scene = None
        
        if scene is not None and scene_index is not None:
            raise ValueError("Provide either scene or scene_index, not both")
        if scene_index is not None:
            if scene_index < 0 or scene_index >= len(scenes):
                raise IndexError(f"Scene index {scene_index} out of range for {file_path}")
            selected_scene = scenes[scene_index]
        elif scene is not None:
            if scene not in scenes:
                raise ValueError(f"Unknown scene '{scene}' for {file_path}")
            selected_scene = scene
        elif scenes:
            selected_scene = scenes[0]
        
        if selected_scene is not None and hasattr(img, "set_scene"):
            img.set_scene(selected_scene)
        
        dims_order = str(getattr(img.dims, "order", ""))
        if "S" in dims_order:
            raw = img.get_image_data("TCZYX", S=0)
        else:
            raw = img.get_image_data("TCZYX")
        
        data, axes = SmartFileLoader._squeeze_singleton_axes(np.asarray(raw), ("T", "C", "Z", "Y", "X"))
        if data.ndim == 0:
            data = data.reshape((1,))
            axes = ("Y",)
        
        pps = img.physical_pixel_sizes
        axis_scale = {
            "T": 1.0,
            "C": 1.0,
            "Z": float(pps.Z) if pps.Z is not None else 1.0,
            "Y": float(pps.Y) if pps.Y is not None else 1.0,
            "X": float(pps.X) if pps.X is not None else 1.0,
        }
        scale = [float(axis_scale.get(axis, 1.0)) for axis in axes]
        channel_names = [str(name) for name in (img.channel_names or [])]
        if "C" not in axes:
            channel_names = []
        
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': 'aicsimageio',
            'scale': scale,
            'axes': list(axes),
            'scene': selected_scene,
            'available_scenes': scenes,
            'channel_names': channel_names,
            'dims_order': dims_order,
        }
        
        return data, metadata

    @staticmethod
    def _squeeze_singleton_axes(data, axes):
        """Drop singleton dimensions while preserving axis labels."""
        squeezed = np.asarray(data)
        axis_labels = list(axes)
        for idx in reversed(range(len(axis_labels))):
            if idx < squeezed.ndim and squeezed.shape[idx] == 1:
                squeezed = np.squeeze(squeezed, axis=idx)
                axis_labels.pop(idx)
        if not axis_labels:
            axis_labels = ["Y"]
        return squeezed, tuple(axis_labels)
    
    @staticmethod
    def _load_with_pims(file_path):
        """Load with pims"""
        import pims
        
        frames = pims.open(str(file_path))
        
        if len(frames) == 1:
            data = np.array(frames[0])
        else:
            # Multiple frames - stack them
            data = np.array([np.array(frame) for frame in frames])
        
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': 'pims',
            'scale': [1.0] * data.ndim
        }
        
        return data, metadata
    
    @staticmethod
    def _load_with_imageio(file_path):
        """Load with imageio"""
        import imageio
        
        data = imageio.imread(file_path)
        
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': 'imageio',
            'scale': [1.0] * data.ndim
        }
        
        return data, metadata
    
    @staticmethod
    def _load_with_pillow(file_path):
        """Load with PIL/Pillow"""
        from PIL import Image as PILImage
        
        img = PILImage.open(file_path)
        data = np.array(img)
        
        metadata = {
            'name': file_path.stem,
            'file_path': str(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'reader': 'pillow',
            'scale': [1.0] * data.ndim
        }
        
        return data, metadata

class FileLoadThread(QThread):
    """Thread for loading image files without blocking UI"""
    progress = pyqtSignal(int)
    finished_load = pyqtSignal(object, dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, load_options=None):
        super().__init__()
        self.file_path = file_path
        self.load_options = load_options or {}
        
    def run(self):
        """Load image file in background thread"""
        try:
            self.progress.emit(10)
            
            # Use smart loader
            data, metadata = SmartFileLoader.load_image(self.file_path, self.load_options)
            
            self.progress.emit(90)
            
            # Add reader info to metadata
            metadata['load_options'] = self.load_options
            
            self.progress.emit(100)
            self.finished_load.emit(data, metadata)
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to load {self.file_path}: {str(e)}")

class EnhancedFileIOWidget(QWidget):
    """Enhanced File I/O Widget that works without aicsimageio"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.load_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Available readers info
        readers_group = QGroupBox("Available Readers")
        readers_layout = QVBoxLayout()
        
        if available_readers:
            for reader, description in available_readers.items():
                readers_layout.addWidget(QLabel(f"✓ {reader}: {description}"))
        else:
            readers_layout.addWidget(QLabel("⚠ No readers available! Install: pip install tifffile imageio"))
        
        readers_group.setLayout(readers_layout)
        layout.addWidget(readers_group)
        
        # File Import Section
        import_group = QGroupBox("Import Images")
        import_layout = QVBoxLayout()
        
        # Import buttons
        button_layout = QHBoxLayout()
        self.open_file_btn = QPushButton("Open File...")
        self.open_file_btn.clicked.connect(self.open_file_dialog)
        button_layout.addWidget(self.open_file_btn)
        
        self.open_series_btn = QPushButton("Open Series...")
        self.open_series_btn.clicked.connect(self.open_series_dialog)
        button_layout.addWidget(self.open_series_btn)
        
        import_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        import_layout.addWidget(self.progress_bar)
        
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)
        
        # Info display
        info_group = QGroupBox("File Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setPlainText("No file loaded")
        info_layout.addWidget(self.info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def open_file_dialog(self):
        """Open file dialog for single file selection"""
        # Build filter string based on available readers
        filters = ["All Files (*)"]
        
        if 'tifffile' in available_readers:
            filters.insert(0, "TIFF files (*.tif *.tiff *.lsm)")
        if 'czifile' in available_readers:
            filters.insert(0, "Zeiss CZI (*.czi)")
        if 'readlif' in available_readers:
            filters.insert(0, "Leica LIF (*.lif)")
        if 'nd2reader' in available_readers:
            filters.insert(0, "Nikon ND2 (*.nd2)")
        if 'mrcfile' in available_readers:
            filters.insert(0, "MRC files (*.mrc *.map)")
        if 'zarr' in available_readers:
            filters.insert(0, "Zarr arrays (*.zarr)")
        if 'h5py' in available_readers:
            filters.insert(0, "Imaris files (*.ims);;HDF5 files (*.h5 *.hdf5)")
        if 'imageio' in available_readers or 'pillow' in available_readers:
            filters.insert(0, "Common formats (*.png *.jpg *.jpeg *.bmp)")
        
        filter_string = ";;".join(filters)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            filter_string
        )
        
        if file_path:
            load_options = self._select_scene_load_options(file_path)
            if load_options is None:
                return
            self.load_image_file(file_path, load_options=load_options)
            
    def open_series_dialog(self):
        """Open directory dialog for image series"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory with Image Series"
        )
        
        if directory:
            self.load_image_series(directory)
            
    def load_image_file(self, file_path, load_options=None):
        """Load a single image file"""
        if not Path(file_path).exists():
            self.show_error(f"File not found: {file_path}")
            return
            
        # Start loading in background thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.open_file_btn.setEnabled(False)
        
        self.load_thread = FileLoadThread(file_path, load_options=load_options)
        self.load_thread.progress.connect(self.progress_bar.setValue)
        self.load_thread.finished_load.connect(self.on_image_loaded)
        self.load_thread.error_occurred.connect(self.on_load_error)
        self.load_thread.start()

    def _select_scene_load_options(self, file_path):
        """Prompt for a scene when a file exposes multiple scenes."""
        extension = Path(file_path).suffix.lower()
        if extension not in {".ims", ".czi", ".lif", ".nd2", ".oib", ".oif"}:
            return {}
        try:
            from pymaris.io import list_scenes
        except Exception:
            return {}

        try:
            scenes = list_scenes(file_path)
        except Exception:
            return {}
        if len(scenes) <= 1:
            return {}

        selected, ok = QInputDialog.getItem(
            self,
            "Select Scene",
            "This file contains multiple scenes:",
            scenes,
            0,
            False,
        )
        if not ok:
            return None
        if not selected:
            return {}
        return {"scene": str(selected)}
        
    def load_image_series(self, directory):
        """Load image series from directory"""
        try:
            directory = Path(directory)
            
            # Find supported files
            supported_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
            image_files = []
            
            for ext in supported_extensions:
                image_files.extend(list(directory.glob(f"*{ext}")))
                
            if not image_files:
                self.show_error("No supported image files found in directory")
                return
                
            image_files = sorted(image_files)
            
            # Load first few to create stack
            if 'tifffile' in available_readers:
                import tifffile
                
                # Read first image to get shape and dtype
                first_img = tifffile.imread(image_files[0])
                
                # Check if first image is already 3D (e.g. multi-page TIFF)
                # If so, we might be loading a series of stacks (4D) or just one stack
                
                try:
                    # Create stack
                    # Note: This assumes all images have the same shape
                    stack_shape = (len(image_files),) + first_img.shape
                    stack_data = np.zeros(stack_shape, dtype=first_img.dtype)
                    stack_data[0] = first_img
                    
                    # Load remaining
                    for i, file_path in enumerate(image_files[1:], 1):
                        img = tifffile.imread(file_path)
                        if img.shape != first_img.shape:
                            print(f"Warning: Shape mismatch in series. Expected {first_img.shape}, got {img.shape} for {file_path.name}")
                            # Try to handle mismatch if it's just a missing dimension (e.g. (1, Y, X) vs (Y, X))
                            if img.ndim == first_img.ndim - 1:
                                img = np.expand_dims(img, axis=0)
                            elif img.ndim == first_img.ndim + 1:
                                img = np.squeeze(img)
                            
                            if img.shape != first_img.shape:
                                raise ValueError(f"Shape mismatch: {first_img.shape} vs {img.shape}")
                        
                        stack_data[i] = img
                except ValueError as e:
                    self.show_error(f"Failed to load series due to shape mismatch: {str(e)}")
                    return
                except Exception as e:
                    self.show_error(f"Error loading series: {str(e)}")
                    return
                    
            else:
                self.show_error("tifffile required for series loading")
                return
                
            # Add to viewer
            self.viewer.add_image(
                stack_data,
                name=f"Series_{directory.name}",
                scale=[1.0] * stack_data.ndim
            )
            
            self.info_text.setPlainText(f"Loaded series: {len(image_files)} images from {directory.name}")
            print(f"Loaded image series: {len(image_files)} files")
            
        except Exception as e:
            self.show_error(f"Failed to load series: {str(e)}")
            
    def on_image_loaded(self, data, metadata):
        """Handle successful image loading"""
        try:
            # Add image to viewer
            layer = self.viewer.add_image(
                data,
                name=metadata['name'],
                scale=metadata.get('scale', [1.0] * data.ndim),
                metadata=metadata
            )
            
            # Update info display
            info_text = f"Loaded: {metadata['name']}\n"
            info_text += f"Shape: {metadata['shape']}\n"
            info_text += f"Data type: {metadata['dtype']}\n"
            info_text += f"Reader: {metadata['reader']}"
            
            self.info_text.setPlainText(info_text)
            
            print(f"Successfully loaded: {metadata['name']} with {metadata['reader']}")
            
        except Exception as e:
            self.show_error(f"Failed to add image to viewer: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.open_file_btn.setEnabled(True)
            
    def on_load_error(self, error_message):
        """Handle loading errors"""
        self.show_error(error_message)
        self.progress_bar.setVisible(False)
        self.open_file_btn.setEnabled(True)
        
    def show_error(self, message):
        """Show error message"""
        print(f"Error: {message}")
        self.info_text.setPlainText(f"Error: {message}")

# Make this the default FileIOWidget
FileIOWidget = EnhancedFileIOWidget
