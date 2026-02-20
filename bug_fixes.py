"""
Critical Bug Fixes for PyMaris Scientific Image Analyzer
Addresses issues identified in comprehensive debugging report
"""

import getpass
import os
from pathlib import Path
from textwrap import dedent


def _normalize_generated_block(text: str) -> str:
    """Return normalized generated file text with predictable leading/trailing newlines."""
    return dedent(text).lstrip("\n").rstrip() + "\n"


def _build_track_endpoint_code() -> str:
    """Return fixed Flask `/track` endpoint code snippet for legacy patch workflows."""
    return _normalize_generated_block(
        """
        def track_objects():
            \"\"\"Track objects over time using real timelapse data\"\"\"
            data = request.get_json()
            session_id = data.get('session_id')

            if session_id not in analysis_cache:
                return jsonify({'error': 'Invalid session'}), 400

            try:
                # Check if we have timelapse data
                session_data = analysis_cache[session_id]

                if 'timelapse_data' in session_data:
                    # Use real timelapse processor
                    from timelapse_processor import TimelapseProcessor
                    processor = TimelapseProcessor()

                    # Process timelapse sequence
                    sequence = session_data['timelapse_data']
                    tracking_results = processor.track_objects_in_sequence(sequence)

                    return jsonify({
                        'success': True,
                        'tracks': tracking_results,
                        'num_tracks': len(tracking_results),
                        'num_frames': len(sequence)
                    })
                else:
                    # No timelapse data available
                    return jsonify({
                        'error': 'No timelapse data available for tracking',
                        'suggestion': 'Upload multiple images for time series analysis'
                    }), 400

            except Exception as e:
                return jsonify({'error': f'Tracking failed: {str(e)}'}), 500
        """
    )


def _build_analysis_utils_module() -> str:
    """Return standalone `analysis_utils.py` module text with dependency fallbacks."""
    return _normalize_generated_block(
        """
        \"\"\"
        Analysis Utilities with Dependency Fallbacks
        Provides scientific analysis functions with pure Python fallbacks
        \"\"\"

        # Check for optional dependencies
        try:
            import numpy as np
            HAS_NUMPY = True
        except ImportError:
            HAS_NUMPY = False
            np = None

        try:
            import scipy
            from scipy import ndimage, stats
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False
            scipy = None

        try:
            import pandas as pd
            HAS_PANDAS = True
        except ImportError:
            HAS_PANDAS = False
            pd = None

        try:
            from skimage import filters, measure, segmentation
            HAS_SKIMAGE = True
        except ImportError:
            HAS_SKIMAGE = False

        def calculate_statistics(data):
            \"\"\"Calculate basic statistics with fallbacks\"\"\"
            if HAS_NUMPY and isinstance(data, np.ndarray):
                return {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                }
            else:
                # Pure Python fallback
                flat_data = []
                if isinstance(data, list):
                    for row in data:
                        if isinstance(row, list):
                            flat_data.extend(row)
                        else:
                            flat_data.append(row)
                else:
                    flat_data = list(data)

                if not flat_data:
                    return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

                mean_val = sum(flat_data) / len(flat_data)
                variance = sum((x - mean_val) ** 2 for x in flat_data) / len(flat_data)
                std_val = variance ** 0.5

                return {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min(flat_data),
                    'max': max(flat_data)
                }

        def correlation_coefficient(x, y):
            \"\"\"Calculate correlation coefficient with fallbacks\"\"\"
            if HAS_SCIPY:
                from scipy.stats import pearsonr
                corr, _ = pearsonr(x, y)
                return corr
            else:
                # Pure Python implementation
                if len(x) != len(y) or len(x) == 0:
                    return 0

                mean_x = sum(x) / len(x)
                mean_y = sum(y) / len(y)

                numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
                sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
                sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))

                denominator = (sum_sq_x * sum_sq_y) ** 0.5

                if denominator == 0:
                    return 0

                return numerator / denominator

        def create_dataframe(data):
            \"\"\"Create DataFrame or return list based on pandas availability\"\"\"
            if HAS_PANDAS:
                return pd.DataFrame(data)
            else:
                return data

        def gaussian_filter(image, sigma):
            \"\"\"Apply Gaussian filter with fallbacks\"\"\"
            if HAS_SCIPY:
                return ndimage.gaussian_filter(image, sigma)
            else:
                # Simple blur fallback
                return simple_blur(image, int(sigma))

        def simple_blur(image, radius):
            \"\"\"Simple blur implementation\"\"\"
            try:
                height = len(image)
                width = len(image[0]) if height > 0 else 0

                blurred = [[0 for _ in range(width)] for _ in range(height)]

                for y in range(height):
                    for x in range(width):
                        total = 0
                        count = 0

                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    total += image[ny][nx]
                                    count += 1

                        blurred[y][x] = total / count if count > 0 else 0

                return blurred

            except Exception as e:
                print(f"Blur error: {e}")
                return image
        """
    )


def _build_fixed_requirements_text() -> str:
    """Return requirements template generated by the legacy bug-fix bootstrap."""
    return _normalize_generated_block(
        """
        # PyMaris Scientific Image Analyzer - Fixed Requirements

        # CORE DEPENDENCIES (Required for basic functionality)
        # None - Application works with pure Python fallbacks

        # OPTIONAL SCIENTIFIC COMPUTING
        # Install for enhanced analysis capabilities
        numpy>=1.21.0
        scipy>=1.7.0
        scikit-image>=0.18.0
        matplotlib>=3.5.0

        # OPTIONAL IMAGE I/O
        # Install for comprehensive file format support
        tifffile>=2021.0.0  # For proper TIFF export
        mrcfile>=1.3.0      # For proper MRC export (ChimeraX compatibility)
        imageio>=2.9.0
        Pillow>=8.0.0
        opencv-python>=4.5.0

        # OPTIONAL WEB FRAMEWORK
        # Flask-based interface (alternative to built-in server)
        flask>=2.0.0
        werkzeug>=2.0.0

        # OPTIONAL DATA ANALYSIS
        pandas>=1.3.0

        # NAPARI DESKTOP VERSION (Optional)
        # Full desktop application with plugin ecosystem
        napari[all]>=0.4.15
        magicgui>=0.5.0
        qtpy

        # MICROSCOPY FILE FORMATS (Optional)
        aicsimageio>=4.0.0
        readlif>=0.6.0
        pylibczirw>=3.0.0
        czifile>=2019.0.0
        nd2reader>=3.2.0

        # AI/ML FEATURES (Optional)
        cellpose>=2.0.0
        stardist>=0.8.0
        btrack>=0.4.0

        # DEVELOPMENT TOOLS (Optional)
        pytest>=6.0.0
        black>=21.0.0
        flake8>=3.9.0
        """
    )


class TIFFExportFix:
    """Fix non-standard TIFF export formats for scientific software compatibility"""

    @staticmethod
    def export_proper_tiff(labels, filepath):
        """Export segmentation mask as proper TIFF file"""
        try:
            # Try using tifffile library for proper TIFF format
            try:
                import numpy as np
                import tifffile

                # Convert labels to numpy array if needed
                if isinstance(labels, list):
                    labels_array = np.array(labels, dtype=np.uint16)
                else:
                    labels_array = labels.astype(np.uint16)

                # Write proper TIFF file
                tifffile.imwrite(filepath, labels_array,
                                metadata={'description': 'PyMaris segmentation mask'})
                return True, "Standard TIFF"

            except ImportError:
                # Fallback: Export as ImageJ-compatible text
                TIFFExportFix.export_imagej_text(labels, filepath.replace('.tiff', '.txt'))
                return True, "ImageJ Text"

        except Exception as e:
            print(f"TIFF export error: {e}")
            return False, str(e)

    @staticmethod
    def export_imagej_text(labels, filepath):
        """Export as ImageJ-compatible text format"""
        with open(filepath, 'w') as f:
            f.write("# ImageJ compatible segmentation mask\n")
            f.write(f"# Width: {len(labels[0]) if labels else 0}\n")
            f.write(f"# Height: {len(labels)}\n")
            f.write("# Import: File > Import > Text Image...\n")

            for row in labels:
                f.write('\t'.join(map(str, row)) + '\n')

class MRCExportFix:
    """Fix MRC export format for ChimeraX compatibility"""

    @staticmethod
    def export_proper_mrc(volume_data, filepath):
        """Export volume data as proper MRC file"""
        try:
            # Try using mrcfile library
            try:
                import mrcfile
                import numpy as np

                if isinstance(volume_data, list):
                    volume_array = np.array(volume_data, dtype=np.float32)
                else:
                    volume_array = volume_data.astype(np.float32)

                with mrcfile.new(filepath, overwrite=True) as mrc:
                    mrc.set_data(volume_array)
                    mrc.header.map = mrcfile.constants.MAP_ID
                    mrc.header.machst = mrcfile.utils.machine_stamp()
                    mrc.update_header_from_data()
                    mrc.update_header_stats()

                return True, "Standard MRC"

            except ImportError:
                # Fallback: Export as raw binary with header
                MRCExportFix.export_raw_volume(volume_data, filepath.replace('.mrc', '.raw'))
                return True, "Raw Binary"

        except Exception as e:
            print(f"MRC export error: {e}")
            return False, str(e)

    @staticmethod
    def export_raw_volume(volume_data, filepath):
        """Export as raw binary volume with metadata file"""
        try:
            import numpy as np

            if isinstance(volume_data, list):
                # Convert nested list to flat array
                flat_data = []
                for slice_2d in volume_data:
                    for row in slice_2d:
                        for pixel in row:
                            flat_data.append(float(pixel))
                volume_array = np.array(flat_data, dtype=np.float32)
                shape = (len(volume_data), len(volume_data[0]), len(volume_data[0][0]))
            else:
                volume_array = volume_data.astype(np.float32)
                shape = volume_array.shape

            # Write binary data
            with open(filepath, 'wb') as f:
                volume_array.tobytes()

            # Write metadata file
            meta_filepath = filepath.replace('.raw', '_meta.txt')
            with open(meta_filepath, 'w') as f:
                f.write("# PyMaris Volume Data\n")
                f.write(f"Dimensions: {' x '.join(map(str, shape))}\n")
                f.write("Data type: float32\n")
                f.write("Byte order: little-endian\n")
                f.write("Import instructions: Use as raw volume in visualization software\n")

        except Exception as e:
            print(f"Raw volume export error: {e}")

class ChimeraXPathFix:
    """Fix hardcoded ChimeraX installation paths"""

    @staticmethod
    def find_chimerax_installation():
        """Find ChimeraX installation with expanded search"""
        # Common installation paths
        common_paths = [
            # Windows
            r"C:\Program Files\ChimeraX\bin\ChimeraX.exe",
            r"C:\Program Files (x86)\ChimeraX\bin\ChimeraX.exe",
            r"C:\Users\{username}\AppData\Local\ChimeraX\bin\ChimeraX.exe",
            # macOS
            "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
            "/usr/local/bin/chimerax",
            # Linux
            "/usr/bin/chimerax",
            "/usr/local/bin/chimerax",
            "/opt/chimerax/bin/chimerax",
            "~/bin/chimerax",
            "~/.local/bin/chimerax"
        ]

        # Check environment variable first
        chimera_path = os.environ.get('CHIMERAX_PATH')
        if chimera_path and os.path.exists(chimera_path):
            return chimera_path

        # Check common paths
        try:
            username = os.getlogin()
        except Exception:
            username = getpass.getuser() or "user"
        for path in common_paths:
            expanded_path = os.path.expanduser(path.replace('{username}', username))
            if os.path.exists(expanded_path):
                return expanded_path

        # Check PATH environment variable
        for path_dir in os.environ.get('PATH', '').split(os.pathsep):
            chimera_exe = os.path.join(path_dir, 'chimerax')
            if os.path.exists(chimera_exe):
                return chimera_exe
            chimera_exe = os.path.join(path_dir, 'ChimeraX.exe')
            if os.path.exists(chimera_exe):
                return chimera_exe

        return None

    @staticmethod
    def create_config_with_chimerax_path():
        """Create configuration file for ChimeraX path"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        config_file = config_dir / "config.json"

        import json

        default_config = {
            "chimerax_path": "",
            "chimerax_auto_detect": True,
            "export_formats": {
                "tiff_use_library": True,
                "mrc_use_library": True
            },
            "performance": {
                "max_workers": 4,
                "memory_limit_mb": 2048
            }
        }

        # Try to detect ChimeraX
        detected_path = ChimeraXPathFix.find_chimerax_installation()
        if detected_path:
            default_config["chimerax_path"] = detected_path

        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)

        return config_file

class TimelapseLogicFix:
    """Fix inconsistent timelapse logic between app.py and timelapse_processor.py"""

    @staticmethod
    def fix_app_track_endpoint():
        """Return proper tracking logic instead of simulation"""
        return _build_track_endpoint_code()

class PhaseCorrelationFix:
    """Fix incomplete phase correlation implementation"""

    @staticmethod
    def implement_phase_correlation(image1, image2):
        """Proper phase correlation alignment implementation"""
        try:
            # Try using scipy for FFT-based phase correlation
            try:
                import numpy as np
                from scipy.fft import fft2, fftshift, ifft2

                # Convert to float
                img1 = np.array(image1, dtype=np.float64)
                img2 = np.array(image2, dtype=np.float64)

                # Compute FFTs
                fft1 = fft2(img1)
                fft2 = fft2(img2)

                # Cross-power spectrum
                cross_power = (fft1 * np.conj(fft2)) / (np.abs(fft1 * np.conj(fft2)) + 1e-10)

                # Inverse FFT to get correlation
                correlation = np.real(ifft2(cross_power))
                correlation = fftshift(correlation)

                # Find peak
                peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)

                # Convert to displacement
                center_y, center_x = np.array(correlation.shape) // 2
                dy = peak_y - center_y
                dx = peak_x - center_x

                return dx, dy, np.max(correlation)

            except ImportError:
                # Fallback to cross-correlation
                return PhaseCorrelationFix.cross_correlation_fallback(image1, image2)

        except Exception as e:
            print(f"Phase correlation error: {e}")
            return 0, 0, 0

    @staticmethod
    def cross_correlation_fallback(image1, image2):
        """Fallback cross-correlation implementation"""
        try:
            best_dx, best_dy, best_score = 0, 0, 0
            search_range = 20  # Limit search for performance

            height1, width1 = len(image1), len(image1[0])
            height2, width2 = len(image2), len(image2[0])

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    score = 0
                    count = 0

                    for y in range(max(0, dy), min(height1, height2 + dy)):
                        for x in range(max(0, dx), min(width1, width2 + dx)):
                            y1, x1 = y, x
                            y2, x2 = y - dy, x - dx

                            if (0 <= y2 < height2 and 0 <= x2 < width2):
                                score += image1[y1][x1] * image2[y2][x2]
                                count += 1

                    if count > 0:
                        normalized_score = score / count
                        if normalized_score > best_score:
                            best_score = normalized_score
                            best_dx, best_dy = dx, dy

            return best_dx, best_dy, best_score

        except Exception as e:
            print(f"Cross-correlation fallback error: {e}")
            return 0, 0, 0

class DependencyFallbackFix:
    """Fix missing dependency fallbacks in utilities"""

    @staticmethod
    def create_analysis_utils_with_fallbacks():
        """Create analysis_utils.py with proper fallbacks"""
        return _build_analysis_utils_module()

def apply_all_fixes():
    """Apply all critical bug fixes"""
    fixes_applied = []

    try:
        # Create config directory and ChimeraX configuration
        config_file = ChimeraXPathFix.create_config_with_chimerax_path()
        fixes_applied.append(f"Created configuration file: {config_file}")

        # Create proper analysis utilities
        utils_dir = Path("utils")
        utils_dir.mkdir(exist_ok=True)

        utils_file = utils_dir / "analysis_utils.py"
        with open(utils_file, 'w') as f:
            f.write(DependencyFallbackFix.create_analysis_utils_with_fallbacks())
        fixes_applied.append(f"Created analysis utilities with fallbacks: {utils_file}")

        # Create updated requirements.txt
        req_file = Path("requirements_fixed.txt")
        with open(req_file, 'w') as f:
            f.write(_build_fixed_requirements_text())
        fixes_applied.append(f"Created fixed requirements file: {req_file}")

        return fixes_applied

    except Exception as e:
        return [f"Error applying fixes: {e}"]

if __name__ == "__main__":
    print("Applying critical bug fixes...")
    results = apply_all_fixes()
    for result in results:
        print(f"✓ {result}")
    print("Bug fixes completed!")
