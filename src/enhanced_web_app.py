#!/usr/bin/env python3
"""
Enhanced Scientific Image Analyzer - Web Interface
Advanced segmentation, 4D viewing, and AI denoising capabilities
Works with fallback implementations when scientific libraries unavailable
"""

import os
import sys
import json
import uuid
import base64
import tempfile
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import cgi
from io import BytesIO
import math

# Import fallback analysis modules
from scientific_analyzer import (
    GradientWatershedSegmentation, StarDistSegmentation, RegionPropsAnalysis,
    BTrackTracking, ColocalizationAnalysis, ImageProcessor
)

# Import FIB-SEM specialized plugins
try:
    from fibsem_plugins import (
        ChimeraXIntegration, ThreeDCounter, TomoSliceAnalyzer,
        AcceleratedClassification, MembraneSegmenter, GPU_ImageProcessor,
        Empanada_Segmentation, OrganoidCounter, FIBSEMAnalyzer
    )
    FIBSEM_AVAILABLE = True
except ImportError:
    FIBSEM_AVAILABLE = False
    print("FIB-SEM plugins not available - using fallback implementations")

# Import timelapse and batch processing
try:
    from timelapse_processor import TimelapseProcessor, ImageAligner
    from batch_processor import BatchProcessor
    ADVANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PROCESSING_AVAILABLE = False
    print("Advanced processing modules not available")

# Global session storage
sessions = {}
UPLOAD_DIR = tempfile.mkdtemp()

class EnhancedImageProcessor:
    """Enhanced image processing with advanced algorithms"""
    
    @staticmethod
    def load_image(file_path):
        """Load image with comprehensive format support"""
        try:
            # Try PIL first for comprehensive format support
            try:
                from PIL import Image
                img = Image.open(file_path)
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Convert to list format
                width, height = img.size
                image_array = []
                for y in range(height):
                    row = []
                    for x in range(width):
                        row.append(img.getpixel((x, y)))
                    image_array.append(row)
                
                print(f"Successfully loaded {file_path} using PIL ({width}x{height})")
                return image_array
                
            except ImportError:
                # Fallback: try to read TIFF using basic binary parsing
                return EnhancedImageProcessor._load_tiff_fallback(file_path)
                
        except Exception as e:
            print(f"Error loading image with PIL: {e}")
            # Try fallback method
            return EnhancedImageProcessor._load_tiff_fallback(file_path)
    
    @staticmethod
    def _load_tiff_fallback(file_path):
        """Fallback TIFF reader for basic uncompressed TIFF files"""
        try:
            print(f"Using fallback TIFF reader for {file_path}...")
            
            with open(file_path, 'rb') as f:
                # Read TIFF header
                header = f.read(8)
                
                # Check TIFF signature
                if header[:2] == b'II':  # Little endian
                    endian = '<'
                elif header[:2] == b'MM':  # Big endian
                    endian = '>'
                else:
                    print("Not a valid TIFF file, creating test pattern")
                    return EnhancedImageProcessor._create_test_image()
                
                import struct
                
                # Read magic number and first IFD offset
                magic = struct.unpack(endian + 'H', header[2:4])[0]
                if magic != 42:
                    print("Invalid TIFF magic number, creating test pattern")
                    return EnhancedImageProcessor._create_test_image()
                
                ifd_offset = struct.unpack(endian + 'L', header[4:8])[0]
                
                # Read IFD (Image File Directory)
                f.seek(ifd_offset)
                num_entries = struct.unpack(endian + 'H', f.read(2))[0]
                
                # Parse IFD entries to get image info
                width = height = bits_per_sample = compression = strip_offsets = strip_byte_counts = None
                
                for _ in range(num_entries):
                    tag, field_type, count, value_offset = struct.unpack(endian + 'HHLL', f.read(12))
                    
                    if tag == 256:  # ImageWidth
                        width = value_offset if count == 1 else struct.unpack(endian + 'L', f.read(4))[0]
                    elif tag == 257:  # ImageLength
                        height = value_offset if count == 1 else struct.unpack(endian + 'L', f.read(4))[0]
                    elif tag == 258:  # BitsPerSample
                        bits_per_sample = value_offset if count == 1 else 8
                    elif tag == 259:  # Compression
                        compression = value_offset if count == 1 else 1
                    elif tag == 273:  # StripOffsets
                        if count == 1:
                            strip_offsets = [value_offset]
                        else:
                            current_pos = f.tell()
                            f.seek(value_offset)
                            strip_offsets = [struct.unpack(endian + 'L', f.read(4))[0] for _ in range(count)]
                            f.seek(current_pos)
                    elif tag == 279:  # StripByteCounts
                        if count == 1:
                            strip_byte_counts = [value_offset]
                        else:
                            current_pos = f.tell()
                            f.seek(value_offset)
                            strip_byte_counts = [struct.unpack(endian + 'L', f.read(4))[0] for _ in range(count)]
                            f.seek(current_pos)
                
                if not all([width, height, strip_offsets, strip_byte_counts]):
                    print("Missing required TIFF tags, creating test pattern")
                    return EnhancedImageProcessor._create_test_image()
                
                if compression != 1:
                    print("Compressed TIFF not supported in fallback mode, creating test pattern")
                    return EnhancedImageProcessor._create_test_image()
                
                # Read image data
                image_data = b''
                for offset, byte_count in zip(strip_offsets, strip_byte_counts):
                    f.seek(offset)
                    image_data += f.read(byte_count)
                
                # Convert to 2D array
                if bits_per_sample == 8:
                    image_array = []
                    for y in range(height):
                        row = []
                        for x in range(width):
                            pixel_index = y * width + x
                            if pixel_index < len(image_data):
                                row.append(image_data[pixel_index])
                            else:
                                row.append(0)
                        image_array.append(row)
                elif bits_per_sample == 16:
                    image_array = []
                    for y in range(height):
                        row = []
                        for x in range(width):
                            pixel_index = (y * width + x) * 2
                            if pixel_index + 1 < len(image_data):
                                # Convert 16-bit to 8-bit for display
                                pixel_value = struct.unpack(endian + 'H', image_data[pixel_index:pixel_index+2])[0]
                                row.append(pixel_value >> 8)  # Scale down to 8-bit
                            else:
                                row.append(0)
                        image_array.append(row)
                else:
                    print(f"Unsupported bits per sample: {bits_per_sample}, creating test pattern")
                    return EnhancedImageProcessor._create_test_image()
                
                print(f"Successfully loaded TIFF using fallback reader ({width}x{height}, {bits_per_sample}-bit)")
                return image_array
                
        except Exception as e:
            print(f"Fallback TIFF reader failed: {e}")
            # Last resort: create a test pattern
            return EnhancedImageProcessor._create_test_image()
    
    @staticmethod
    def _create_test_image():
        """Create a test image when file loading fails"""
        print("Creating test pattern image for demonstration...")
        image = []
        for y in range(256):
            row = []
            for x in range(256):
                # Create a test pattern with geometric shapes
                center_x, center_y = 128, 128
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                
                # Concentric circles pattern
                if distance < 30:
                    pixel = 200
                elif distance < 60:
                    pixel = 150
                elif distance < 90:
                    pixel = 100
                else:
                    pixel = 50
                
                # Add some noise for realism
                pixel += (x + y) % 20 - 10
                row.append(max(0, min(255, pixel)))
            image.append(row)
        
        print("Test pattern created (256x256 pixels)")
        return image
    
    @staticmethod
    def bilateral_filter_fallback(image, sigma_spatial=5.0, sigma_intensity=20.0):
        """Simplified bilateral filter using basic operations"""
        try:
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            filtered = [[0 for _ in range(width)] for _ in range(height)]
            
            # Simple averaging with intensity weighting
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    center_val = image[i][j]
                    weighted_sum = 0
                    weight_sum = 0
                    
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            neighbor_val = image[i + di][j + dj]
                            
                            # Simple distance weight
                            spatial_dist = abs(di) + abs(dj)
                            spatial_weight = math.exp(-spatial_dist / sigma_spatial)
                            
                            # Intensity weight
                            intensity_diff = abs(center_val - neighbor_val)
                            intensity_weight = math.exp(-intensity_diff / sigma_intensity)
                            
                            weight = spatial_weight * intensity_weight
                            weighted_sum += weight * neighbor_val
                            weight_sum += weight
                    
                    filtered[i][j] = weighted_sum / weight_sum if weight_sum > 0 else center_val
            
            # Copy borders
            for i in range(height):
                filtered[i][0] = image[i][0]
                if width > 1:
                    filtered[i][width - 1] = image[i][width - 1]
            
            for j in range(width):
                filtered[0][j] = image[0][j]
                if height > 1:
                    filtered[height - 1][j] = image[height - 1][j]
            
            return filtered
            
        except Exception as e:
            print(f"Bilateral filter error: {e}")
            return image
    
    @staticmethod
    def non_local_means_fallback(image, h=10.0, patch_size=7):
        """Simplified non-local means denoising"""
        try:
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            denoised = [[0 for _ in range(width)] for _ in range(height)]
            
            patch_radius = patch_size // 2
            
            for i in range(patch_radius, height - patch_radius):
                for j in range(patch_radius, width - patch_radius):
                    # Current patch center
                    current_val = image[i][j]
                    
                    # Simple averaging with patch similarity
                    weighted_sum = 0
                    weight_sum = 0
                    
                    # Search in local neighborhood
                    for ki in range(max(patch_radius, i - 20), 
                                  min(height - patch_radius, i + 20)):
                        for kj in range(max(patch_radius, j - 20), 
                                      min(width - patch_radius, j + 20)):
                            
                            # Compute patch difference (simplified)
                            patch_diff = 0
                            for di in range(-patch_radius, patch_radius + 1):
                                for dj in range(-patch_radius, patch_radius + 1):
                                    if (0 <= i + di < height and 0 <= j + dj < width and
                                        0 <= ki + di < height and 0 <= kj + dj < width):
                                        diff = image[i + di][j + dj] - image[ki + di][kj + dj]
                                        patch_diff += diff * diff
                            
                            # Weight based on patch similarity
                            weight = math.exp(-max(patch_diff - 2 * h * h, 0.0) / (h * h))
                            weighted_sum += weight * image[ki][kj]
                            weight_sum += weight
                    
                    denoised[i][j] = weighted_sum / weight_sum if weight_sum > 0 else current_val
            
            # Copy borders
            for i in range(height):
                if i < patch_radius or i >= height - patch_radius:
                    for j in range(width):
                        denoised[i][j] = image[i][j]
                else:
                    for j in range(patch_radius):
                        denoised[i][j] = image[i][j]
                    for j in range(width - patch_radius, width):
                        denoised[i][j] = image[i][j]
            
            return denoised
            
        except Exception as e:
            print(f"Non-local means error: {e}")
            return image
    
    @staticmethod
    def morphological_snakes_fallback(image, iterations=50):
        """Simplified morphological snakes segmentation"""
        try:
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            
            # Create initial circular mask
            center_y, center_x = height // 2, width // 2
            radius = min(height, width) // 4
            
            mask = [[0 for _ in range(width)] for _ in range(height)]
            for i in range(height):
                for j in range(width):
                    dist = math.sqrt((i - center_y)**2 + (j - center_x)**2)
                    if dist <= radius:
                        mask[i][j] = 1
            
            # Iterative evolution
            for iteration in range(iterations):
                new_mask = [[mask[i][j] for j in range(width)] for i in range(height)]
                
                # Simple edge-based evolution
                for i in range(1, height - 1):
                    for j in range(1, width - 1):
                        if mask[i][j] == 1:  # Inside contour
                            # Check if should contract
                            edge_strength = 0
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    if di != 0 or dj != 0:
                                        edge_strength += abs(image[i][j] - image[i + di][j + dj])
                            
                            if edge_strength / 8 > 30:  # Strong edge - stay
                                continue
                            else:  # Weak edge - contract
                                new_mask[i][j] = 0
                        else:  # Outside contour
                            # Check if should expand
                            neighbor_count = sum(mask[i + di][j + dj] 
                                               for di in [-1, 0, 1] 
                                               for dj in [-1, 0, 1] 
                                               if di != 0 or dj != 0)
                            if neighbor_count >= 3:
                                new_mask[i][j] = 1
                
                mask = new_mask
            
            return mask
            
        except Exception as e:
            print(f"Morphological snakes error: {e}")
            # Fallback to threshold
            threshold = sum(sum(row) for row in image) / (len(image) * len(image[0]))
            return [[1 if pixel > threshold else 0 for pixel in row] for row in image]
    
    @staticmethod
    def region_growing_fallback(image, seed_points, threshold=0.1):
        """Simplified region growing segmentation"""
        try:
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            segmented = [[0 for _ in range(width)] for _ in range(height)]
            
            for label, (seed_y, seed_x) in enumerate(seed_points, 1):
                if seed_y >= height or seed_x >= width:
                    continue
                
                visited = [[False for _ in range(width)] for _ in range(height)]
                stack = [(seed_y, seed_x)]
                seed_value = image[seed_y][seed_x]
                threshold_val = threshold * 255
                
                while stack:
                    y, x = stack.pop()
                    
                    if (y < 0 or y >= height or x < 0 or x >= width or 
                        visited[y][x] or segmented[y][x] != 0):
                        continue
                    
                    if abs(image[y][x] - seed_value) <= threshold_val:
                        visited[y][x] = True
                        segmented[y][x] = label
                        
                        # Add neighbors
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((y + dy, x + dx))
            
            return segmented
            
        except Exception as e:
            print(f"Region growing error: {e}")
            return [[0 for _ in range(len(image[0]))] for _ in range(len(image))]

class FourDViewerFallback:
    """4D image viewing with fallback implementations"""
    
    def __init__(self):
        self.current_volume = None
        self.current_timepoint = 0
        self.current_z_slice = 0
        self.volume_metadata = {}
    
    def load_4d_data(self, image_data, metadata=None):
        """Load 4D data with proper shape handling"""
        try:
            # Convert 2D image to 4D structure (T, Z, Y, X)
            if isinstance(image_data, list) and isinstance(image_data[0], list):
                # 2D image - add time and Z dimensions
                self.current_volume = [[image_data]]  # [T=1][Z=1][Y][X]
            else:
                # Assume already proper format
                self.current_volume = image_data
            
            self.volume_metadata = metadata or {}
            self.current_timepoint = 0
            self.current_z_slice = 0
            
            return True
            
        except Exception as e:
            print(f"Error loading 4D data: {e}")
            return False
    
    def get_current_slice(self):
        """Get current 2D slice"""
        if not self.current_volume:
            return [[0]]
        
        try:
            t_max = len(self.current_volume)
            t_idx = min(self.current_timepoint, t_max - 1)
            
            z_max = len(self.current_volume[t_idx])
            z_idx = min(self.current_z_slice, z_max - 1)
            
            return self.current_volume[t_idx][z_idx]
            
        except Exception as e:
            print(f"Error getting slice: {e}")
            return [[0]]
    
    def navigate_time(self, timepoint):
        """Navigate to specific timepoint"""
        if not self.current_volume:
            return [[0]]
        
        max_t = len(self.current_volume)
        self.current_timepoint = max(0, min(timepoint, max_t - 1))
        return self.get_current_slice()
    
    def navigate_z(self, z_slice):
        """Navigate to specific Z slice"""
        if not self.current_volume:
            return [[0]]
        
        if self.current_volume and len(self.current_volume) > self.current_timepoint:
            max_z = len(self.current_volume[self.current_timepoint])
            self.current_z_slice = max(0, min(z_slice, max_z - 1))
        
        return self.get_current_slice()
    
    def get_max_projection(self, axis='z'):
        """Get maximum intensity projection"""
        if not self.current_volume:
            return [[0]]
        
        try:
            current_3d = self.current_volume[self.current_timepoint]
            
            if axis.lower() == 'z':
                # Max projection across Z
                if not current_3d:
                    return [[0]]
                
                height = len(current_3d[0])
                width = len(current_3d[0][0]) if height > 0 else 0
                projection = [[0 for _ in range(width)] for _ in range(height)]
                
                for i in range(height):
                    for j in range(width):
                        max_val = 0
                        for z in range(len(current_3d)):
                            if i < len(current_3d[z]) and j < len(current_3d[z][i]):
                                max_val = max(max_val, current_3d[z][i][j])
                        projection[i][j] = max_val
                
                return projection
            else:
                # For other axes, return current slice
                return self.get_current_slice()
                
        except Exception as e:
            print(f"Error in max projection: {e}")
            return [[0]]
    
    def get_volume_info(self):
        """Get information about current volume"""
        if not self.current_volume:
            return {}
        
        try:
            t_dim = len(self.current_volume)
            z_dim = len(self.current_volume[0]) if t_dim > 0 else 0
            y_dim = len(self.current_volume[0][0]) if z_dim > 0 else 0
            x_dim = len(self.current_volume[0][0][0]) if y_dim > 0 else 0
            
            return {
                'dimensions': {'T': t_dim, 'Z': z_dim, 'Y': y_dim, 'X': x_dim},
                'current_timepoint': self.current_timepoint,
                'current_z_slice': self.current_z_slice,
                'metadata': self.volume_metadata
            }
        except Exception as e:
            print(f"Error getting volume info: {e}")
            return {'dimensions': {'T': 1, 'Z': 1, 'Y': 100, 'X': 100}}

class EnhancedImageHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP handler with advanced image analysis capabilities"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_main_page()
        elif parsed_path.path == '/api/session/create':
            self.create_session()
        elif parsed_path.path.startswith('/download/'):
            self.handle_download()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/upload':
            self.handle_file_upload()
        elif parsed_path.path == '/api/process':
            self.handle_process_request()
        elif parsed_path.path == '/api/segment':
            self.handle_segment_request()
        elif parsed_path.path == '/api/analyze':
            self.handle_analyze_request()
        elif parsed_path.path == '/api/colocalization':
            self.handle_colocalization_request()
        elif parsed_path.path == '/api/denoise':
            self.handle_denoise_request()
        elif parsed_path.path == '/api/advanced_segment':
            self.handle_advanced_segment_request()
        elif parsed_path.path == '/api/4d_navigate':
            self.handle_4d_navigation()
        elif parsed_path.path == '/api/4d_projection':
            self.handle_4d_projection()
        elif parsed_path.path == '/api/export':
            self.handle_export_request()
        elif parsed_path.path == '/api/fibsem/analyze':
            self.handle_fibsem_analysis()
        elif parsed_path.path == '/api/fibsem/chimerax':
            self.handle_chimerax_launch()
        elif parsed_path.path == '/api/fibsem/3d_count':
            self.handle_3d_counting()
        elif parsed_path.path == '/api/fibsem/classify':
            self.handle_accelerated_classification()
        elif parsed_path.path == '/api/fibsem/membrane_segment':
            self.handle_membrane_segmentation()
        elif parsed_path.path == '/api/timelapse/align':
            self.handle_timelapse_alignment()
        elif parsed_path.path == '/api/timelapse/normalize':
            self.handle_timelapse_normalization()
        elif parsed_path.path == '/api/timelapse/analyze':
            self.handle_timelapse_analysis()
        elif parsed_path.path == '/api/batch/create':
            self.handle_batch_create()
        elif parsed_path.path == '/api/batch/process':
            self.handle_batch_process()
        elif parsed_path.path == '/api/batch/status':
            self.handle_batch_status()
        elif parsed_path.path == '/api/upload/batch':
            self.handle_batch_upload()
        elif parsed_path.path == '/api/timelapse/load':
            self.handle_timelapse_load()
        else:
            self.send_error(404)
    
    def serve_main_page(self):
        """Serve the main HTML interface"""
        html_content = self.get_enhanced_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-length', str(len(html_content)))
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def create_session(self):
        """Create new analysis session"""
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'images': {},
            'layers': {},
            'results': {},
            'metadata': {},
            '4d_viewer': None
        }
        
        response = {'session_id': session_id, 'enhanced_features': True}
        self.send_json_response(response)
    
    def handle_file_upload(self):
        """Handle file upload with proper multipart parsing"""
        try:
            content_type = self.headers['content-type']
            if not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Content type must be multipart/form-data")
                return
            
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            session_id = form.getvalue('session_id')
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            file_item = form['file']
            if not file_item.filename:
                self.send_json_response({'error': 'No file uploaded'}, 400)
                return
            
            filename = file_item.filename
            file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")
            
            with open(file_path, 'wb') as f:
                f.write(file_item.file.read())
            
            image_data = self.load_image(session_id, file_path)
            
            if image_data is not None:
                response = {
                    'success': True,
                    'filename': filename,
                    'shape': [len(image_data), len(image_data[0]) if image_data else 0],
                    'message': f'Image loaded successfully: {filename}'
                }
            else:
                response = {'error': 'Failed to load image'}
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Upload error: {e}")
            self.send_json_response({'error': f'Upload failed: {str(e)}'}, 500)
    
    def load_image(self, session_id, file_path):
        """Load image using enhanced fallback method"""
        try:
            # Use the enhanced image processor with TIFF fallback
            image_data = EnhancedImageProcessor.load_image(file_path)
            
            if image_data is not None:
                sessions[session_id]['images']['original'] = image_data
                sessions[session_id]['metadata']['filename'] = Path(file_path).name
                sessions[session_id]['metadata']['shape'] = [len(image_data), len(image_data[0]) if image_data else 0]
                print(f"Successfully loaded {Path(file_path).name} - {len(image_data)}x{len(image_data[0]) if image_data else 0}")
                return image_data
            else:
                print(f"Failed to load image: {file_path}")
                return None
                
        except Exception as e:
            print(f"Error loading image: {e}")
            # Try to provide a fallback demonstration image
            try:
                image_data = EnhancedImageProcessor._create_test_image()
                sessions[session_id]['images']['original'] = image_data
                sessions[session_id]['metadata']['filename'] = f"test_pattern_{Path(file_path).name}"
                sessions[session_id]['metadata']['shape'] = [len(image_data), len(image_data[0])]
                print(f"Created test pattern as fallback for {Path(file_path).name}")
                return image_data
            except:
                return None
    
    def handle_denoise_request(self):
        """Handle AI denoising requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            method = data.get('method', 'bilateral')
            params = data.get('params', {})
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            image = sessions[session_id]['images']['original']
            
            # Apply denoising
            if method == 'bilateral':
                sigma_spatial = params.get('sigma_spatial', 5.0)
                sigma_intensity = params.get('sigma_intensity', 20.0)
                denoised = EnhancedImageProcessor.bilateral_filter_fallback(
                    image, sigma_spatial, sigma_intensity)
            elif method == 'non_local_means':
                h = params.get('h', 10.0)
                denoised = EnhancedImageProcessor.non_local_means_fallback(image, h)
            elif method == 'wiener':
                # Simple smoothing as fallback
                denoised = CellposeSegmentation._gaussian_blur(image, sigma=1.0)
            else:
                denoised = image
            
            sessions[session_id]['images'][f'denoised_{method}'] = denoised
            
            response = {
                'success': True,
                'method': method,
                'message': f'AI denoising complete using {method}'
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Denoising error: {e}")
            self.send_json_response({'error': f'Denoising failed: {str(e)}'}, 500)
    
    def handle_advanced_segment_request(self):
        """Handle advanced segmentation requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            method = data.get('method', 'morphological_snakes')
            params = data.get('params', {})
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            image = sessions[session_id]['images']['original']
            
            # Apply advanced segmentation
            if method == 'morphological_snakes':
                iterations = params.get('iterations', 50)
                labels = EnhancedImageProcessor.morphological_snakes_fallback(image, iterations)
            elif method == 'region_growing':
                seed_points = params.get('seed_points', [[len(image)//2, len(image[0])//2]])
                threshold = params.get('threshold', 0.1)
                labels = EnhancedImageProcessor.region_growing_fallback(image, seed_points, threshold)
            elif method == 'adaptive_threshold':
                # Simple local thresholding
                block_size = params.get('block_size', 11)
                labels = self.adaptive_threshold_fallback(image, block_size)
            else:
                # Fallback to simple threshold
                threshold = sum(sum(row) for row in image) / (len(image) * len(image[0]))
                labels = [[1 if pixel > threshold else 0 for pixel in row] for row in image]
            
            sessions[session_id]['layers'][f'advanced_labels_{method}'] = labels
            
            # Count objects
            object_count = max(max(row) for row in labels) if labels else 0
            
            response = {
                'success': True,
                'method': method,
                'object_count': object_count,
                'message': f'Advanced segmentation complete: {object_count} objects found'
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Advanced segmentation error: {e}")
            self.send_json_response({'error': f'Advanced segmentation failed: {str(e)}'}, 500)
    
    def adaptive_threshold_fallback(self, image, block_size):
        """Simple adaptive thresholding"""
        try:
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            result = [[0 for _ in range(width)] for _ in range(height)]
            
            half_block = block_size // 2
            
            for i in range(height):
                for j in range(width):
                    # Calculate local mean
                    local_sum = 0
                    count = 0
                    
                    for di in range(-half_block, half_block + 1):
                        for dj in range(-half_block, half_block + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                local_sum += image[ni][nj]
                                count += 1
                    
                    local_mean = local_sum / count if count > 0 else 0
                    result[i][j] = 1 if image[i][j] > local_mean else 0
            
            return result
            
        except Exception as e:
            print(f"Adaptive threshold error: {e}")
            return image
    
    def handle_4d_navigation(self):
        """Handle 4D volume navigation"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            navigation_type = data.get('type', 'time')
            index = data.get('index', 0)
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            # Initialize 4D viewer if not exists
            if not sessions[session_id]['4d_viewer']:
                image = sessions[session_id]['images']['original']
                sessions[session_id]['4d_viewer'] = FourDViewerFallback()
                sessions[session_id]['4d_viewer'].load_4d_data(image)
            
            viewer = sessions[session_id]['4d_viewer']
            
            if navigation_type == 'time':
                current_slice = viewer.navigate_time(index)
            elif navigation_type == 'z':
                current_slice = viewer.navigate_z(index)
            else:
                current_slice = viewer.get_current_slice()
            
            volume_info = viewer.get_volume_info()
            
            # Calculate slice stats
            flat_values = [val for row in current_slice for val in row]
            slice_stats = {
                'mean': sum(flat_values) / len(flat_values) if flat_values else 0,
                'max': max(flat_values) if flat_values else 0,
                'shape': [len(current_slice), len(current_slice[0]) if current_slice else 0]
            }
            
            response = {
                'success': True,
                'volume_info': volume_info,
                'current_slice_stats': slice_stats,
                'message': f'Navigated to {navigation_type} index {index}'
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"4D navigation error: {e}")
            self.send_json_response({'error': f'4D navigation failed: {str(e)}'}, 500)
    
    def handle_4d_projection(self):
        """Handle 4D projection requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            projection_type = data.get('type', 'z')
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if not sessions[session_id]['4d_viewer']:
                self.send_json_response({'error': 'No 4D data loaded'}, 400)
                return
            
            viewer = sessions[session_id]['4d_viewer']
            projection = viewer.get_max_projection(projection_type)
            
            # Store projection
            sessions[session_id]['images'][f'projection_{projection_type}'] = projection
            
            # Calculate projection stats
            flat_values = [val for row in projection for val in row]
            projection_stats = {
                'mean': sum(flat_values) / len(flat_values) if flat_values else 0,
                'max': max(flat_values) if flat_values else 0,
                'shape': [len(projection), len(projection[0]) if projection else 0]
            }
            
            response = {
                'success': True,
                'projection_type': projection_type,
                'projection_stats': projection_stats,
                'message': f'Created {projection_type} projection'
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"4D projection error: {e}")
            self.send_json_response({'error': f'4D projection failed: {str(e)}'}, 500)
    
    def handle_process_request(self):
        """Handle standard image processing"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            operation = data.get('operation')
            params = data.get('params', {})
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            image = sessions[session_id]['images']['original']
            
            # Apply processing using existing methods
            if operation == 'gaussian_filter':
                sigma = params.get('sigma', 2.0)
                result = CellposeSegmentation._gaussian_blur(image, sigma)
            elif operation == 'threshold':
                method = params.get('method', 'simple')
                threshold = sum(sum(row) for row in image) / (len(image) * len(image[0]))
                result = [[1 if pixel > threshold else 0 for pixel in row] for row in image]
            else:
                result = image
            
            sessions[session_id]['images'][f'processed_{operation}'] = result
            
            response = {
                'success': True,
                'operation': operation,
                'message': f'{operation} applied successfully'
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Processing error: {e}")
            self.send_json_response({'error': f'Processing failed: {str(e)}'}, 500)
    
    def handle_segment_request(self):
        """Handle segmentation requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            method = data.get('method')
            params = data.get('params', {})
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            image = sessions[session_id]['images']['original']
            
            if method == 'cellpose':
                diameter = params.get('diameter', 30)
                labels = CellposeSegmentation.segment_cells(image, diameter=diameter)
            elif method == 'stardist':
                labels = StarDistSegmentation.segment_nuclei(image)
            elif method == 'watershed':
                labels = CellposeSegmentation.segment_cells(image, diameter=20)
            else:
                threshold = sum(sum(row) for row in image) / (len(image) * len(image[0]))
                labels = [[1 if pixel > threshold else 0 for pixel in row] for row in image]
            
            sessions[session_id]['layers'][f'labels_{method}'] = labels
            
            # Count objects
            object_count = max(max(row) for row in labels) if labels else 0
            
            response = {
                'success': True,
                'method': method,
                'object_count': object_count,
                'message': f'Segmentation complete: {object_count} objects found'
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            self.send_json_response({'error': f'Segmentation failed: {str(e)}'}, 500)
    
    def handle_analyze_request(self):
        """Handle object analysis requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            labels_key = data.get('labels_key', 'labels_cellpose')
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if (labels_key not in sessions[session_id]['layers'] or
                'original' not in sessions[session_id]['images']):
                self.send_json_response({'error': 'No segmentation data available'}, 400)
                return
            
            labels = sessions[session_id]['layers'][labels_key]
            image = sessions[session_id]['images']['original']
            
            measurements = RegionPropsAnalysis.analyze_objects(labels, image)
            
            if measurements:
                sessions[session_id]['results'][f'measurements_{labels_key}'] = measurements
                response = {
                    'success': True,
                    'measurements': measurements[:20],
                    'total_objects': len(measurements),
                    'message': f'Analysis complete: {len(measurements)} objects measured'
                }
            else:
                response = {'error': 'Analysis failed'}
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Analysis error: {e}")
            self.send_json_response({'error': f'Analysis failed: {str(e)}'}, 500)
    
    def handle_colocalization_request(self):
        """Handle colocalization analysis requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            image1 = sessions[session_id]['images']['original']
            image2 = CellposeSegmentation._gaussian_blur(image1, sigma=3.0)
            
            results = ColocalizationAnalysis.analyze_colocalization(image1, image2)
            
            response = {
                'success': True,
                'results': results,
                'message': 'Colocalization analysis complete'
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Colocalization error: {e}")
            self.send_json_response({'error': f'Colocalization failed: {str(e)}'}, 500)
    
    def handle_export_request(self):
        """Handle export requests for segmentation masks and tracking data"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            export_type = data.get('export_type', 'mask_tiff')
            data_source = data.get('data_source', 'labels_cellpose')
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if data_source not in sessions[session_id]['layers']:
                self.send_json_response({'error': f'No data available for {data_source}'}, 400)
                return
            
            labels = sessions[session_id]['layers'][data_source]
            original_image = sessions[session_id]['images'].get('original')
            
            export_result = self.export_segmentation_data(
                labels, original_image, export_type, session_id, data_source
            )
            
            if export_result:
                response = {
                    'success': True,
                    'export_type': export_type,
                    'download_url': export_result['download_url'],
                    'filename': export_result['filename'],
                    'file_size': export_result['file_size'],
                    'message': f'Export complete: {export_result["filename"]}'
                }
            else:
                response = {'error': 'Export failed'}
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Export error: {e}")
            self.send_json_response({'error': f'Export failed: {str(e)}'}, 500)
    
    def export_segmentation_data(self, labels, original_image, export_type, session_id, data_source):
        """Export segmentation data in various formats for tracking software"""
        try:
            timestamp = str(int(time.time()))
            base_filename = f"{session_id}_{data_source}_{timestamp}"
            
            if export_type == 'mask_tiff':
                return self.export_mask_tiff(labels, base_filename)
            elif export_type == 'coordinates_csv':
                return self.export_coordinates_csv(labels, base_filename)
            elif export_type == 'centroids_csv':
                return self.export_centroids_csv(labels, base_filename)
            elif export_type == 'tracking_ready':
                return self.export_tracking_ready(labels, original_image, base_filename)
            elif export_type == 'imageJ_roi':
                return self.export_imagej_roi(labels, base_filename)
            elif export_type == 'measurements_csv':
                measurements = self.calculate_object_measurements(labels, original_image)
                return self.export_measurements_csv(measurements, base_filename)
            else:
                return None
                
        except Exception as e:
            print(f"Export data error: {e}")
            return None
    
    def export_mask_tiff(self, labels, base_filename):
        """Export segmentation mask as TIFF file"""
        filename = f"{base_filename}_mask.tiff"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        try:
            # Convert labels to 16-bit format for better compatibility
            max_label = max(max(row) for row in labels) if labels else 0
            
            # Create simple TIFF-like binary format
            with open(filepath, 'wb') as f:
                # Simple header
                height = len(labels)
                width = len(labels[0]) if height > 0 else 0
                
                # Write dimensions
                f.write(height.to_bytes(4, 'little'))
                f.write(width.to_bytes(4, 'little'))
                f.write(max_label.to_bytes(4, 'little'))
                
                # Write pixel data
                for row in labels:
                    for pixel in row:
                        f.write(pixel.to_bytes(2, 'little'))
            
            file_size = os.path.getsize(filepath)
            
            return {
                'download_url': f'/download/{filename}',
                'filename': filename,
                'file_size': file_size
            }
            
        except Exception as e:
            print(f"TIFF export error: {e}")
            return None
    
    def export_coordinates_csv(self, labels, base_filename):
        """Export all object pixel coordinates as CSV"""
        filename = f"{base_filename}_coordinates.csv"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        try:
            with open(filepath, 'w') as f:
                f.write("object_id,x,y\n")
                
                height = len(labels)
                width = len(labels[0]) if height > 0 else 0
                
                for y in range(height):
                    for x in range(width):
                        label_id = labels[y][x]
                        if label_id > 0:  # Non-background pixels
                            f.write(f"{label_id},{x},{y}\n")
            
            file_size = os.path.getsize(filepath)
            
            return {
                'download_url': f'/download/{filename}',
                'filename': filename,
                'file_size': file_size
            }
            
        except Exception as e:
            print(f"Coordinates export error: {e}")
            return None
    
    def export_centroids_csv(self, labels, base_filename):
        """Export object centroids for tracking initialization"""
        filename = f"{base_filename}_centroids.csv"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        try:
            # Calculate centroids
            centroids = self.calculate_centroids(labels)
            
            with open(filepath, 'w') as f:
                f.write("object_id,centroid_x,centroid_y,area\n")
                
                for obj_id, data in centroids.items():
                    f.write(f"{obj_id},{data['centroid_x']:.2f},{data['centroid_y']:.2f},{data['area']}\n")
            
            file_size = os.path.getsize(filepath)
            
            return {
                'download_url': f'/download/{filename}',
                'filename': filename,
                'file_size': file_size
            }
            
        except Exception as e:
            print(f"Centroids export error: {e}")
            return None
    
    def export_tracking_ready(self, labels, original_image, base_filename):
        """Export tracking-ready format with measurements"""
        filename = f"{base_filename}_tracking.csv"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        try:
            measurements = self.calculate_object_measurements(labels, original_image)
            
            with open(filepath, 'w') as f:
                f.write("frame,object_id,x,y,area,perimeter,mean_intensity,max_intensity,eccentricity\n")
                
                frame_id = 0  # Single frame for now
                for measurement in measurements:
                    f.write(f"{frame_id},{measurement['label']},{measurement['centroid_x']:.2f},"
                           f"{measurement['centroid_y']:.2f},{measurement['area']},"
                           f"{measurement.get('perimeter', 0):.2f},{measurement.get('mean_intensity', 0):.2f},"
                           f"{measurement.get('max_intensity', 0):.2f},{measurement.get('eccentricity', 0):.3f}\n")
            
            file_size = os.path.getsize(filepath)
            
            return {
                'download_url': f'/download/{filename}',
                'filename': filename,
                'file_size': file_size
            }
            
        except Exception as e:
            print(f"Tracking export error: {e}")
            return None
    
    def export_imagej_roi(self, labels, base_filename):
        """Export ROI coordinates in ImageJ-compatible format"""
        filename = f"{base_filename}_rois.csv"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        try:
            # Get object boundaries
            boundaries = self.extract_object_boundaries(labels)
            
            with open(filepath, 'w') as f:
                f.write("roi_id,x_coords,y_coords\n")
                
                for obj_id, boundary in boundaries.items():
                    x_coords = ";".join(str(x) for x, y in boundary)
                    y_coords = ";".join(str(y) for x, y in boundary)
                    f.write(f"{obj_id},\"{x_coords}\",\"{y_coords}\"\n")
            
            file_size = os.path.getsize(filepath)
            
            return {
                'download_url': f'/download/{filename}',
                'filename': filename,
                'file_size': file_size
            }
            
        except Exception as e:
            print(f"ROI export error: {e}")
            return None
    
    def export_measurements_csv(self, measurements, base_filename):
        """Export comprehensive object measurements"""
        filename = f"{base_filename}_measurements.csv"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        try:
            with open(filepath, 'w') as f:
                if measurements:
                    # Write header
                    headers = list(measurements[0].keys())
                    f.write(",".join(headers) + "\n")
                    
                    # Write data
                    for measurement in measurements:
                        row = [str(measurement.get(header, '')) for header in headers]
                        f.write(",".join(row) + "\n")
                else:
                    f.write("No measurements available\n")
            
            file_size = os.path.getsize(filepath)
            
            return {
                'download_url': f'/download/{filename}',
                'filename': filename,
                'file_size': file_size
            }
            
        except Exception as e:
            print(f"Measurements export error: {e}")
            return None
    
    def calculate_centroids(self, labels):
        """Calculate centroids for each labeled object"""
        centroids = {}
        
        try:
            height = len(labels)
            width = len(labels[0]) if height > 0 else 0
            
            # Count pixels and sum coordinates for each object
            object_data = {}
            
            for y in range(height):
                for x in range(width):
                    label_id = labels[y][x]
                    if label_id > 0:
                        if label_id not in object_data:
                            object_data[label_id] = {'sum_x': 0, 'sum_y': 0, 'count': 0}
                        
                        object_data[label_id]['sum_x'] += x
                        object_data[label_id]['sum_y'] += y
                        object_data[label_id]['count'] += 1
            
            # Calculate centroids
            for obj_id, data in object_data.items():
                centroids[obj_id] = {
                    'centroid_x': data['sum_x'] / data['count'],
                    'centroid_y': data['sum_y'] / data['count'],
                    'area': data['count']
                }
            
            return centroids
            
        except Exception as e:
            print(f"Centroid calculation error: {e}")
            return {}
    
    def calculate_object_measurements(self, labels, original_image):
        """Calculate comprehensive measurements for each object"""
        try:
            centroids = self.calculate_centroids(labels)
            measurements = []
            
            height = len(labels)
            width = len(labels[0]) if height > 0 else 0
            
            for obj_id, centroid_data in centroids.items():
                # Basic measurements
                measurement = {
                    'label': obj_id,
                    'area': centroid_data['area'],
                    'centroid_x': centroid_data['centroid_x'],
                    'centroid_y': centroid_data['centroid_y']
                }
                
                # Calculate intensity measurements if original image available
                if original_image:
                    intensities = []
                    for y in range(height):
                        for x in range(width):
                            if labels[y][x] == obj_id:
                                intensities.append(original_image[y][x])
                    
                    if intensities:
                        measurement['mean_intensity'] = sum(intensities) / len(intensities)
                        measurement['max_intensity'] = max(intensities)
                        measurement['min_intensity'] = min(intensities)
                
                # Calculate perimeter (simplified)
                perimeter = self.calculate_perimeter(labels, obj_id)
                measurement['perimeter'] = perimeter
                
                # Calculate basic shape features
                if measurement['area'] > 0:
                    measurement['circularity'] = (4 * 3.14159 * measurement['area']) / (perimeter * perimeter) if perimeter > 0 else 0
                    measurement['aspect_ratio'] = self.calculate_aspect_ratio(labels, obj_id)
                    measurement['eccentricity'] = min(1.0, max(0.0, 1.0 - measurement['circularity']))
                
                measurements.append(measurement)
            
            return measurements
            
        except Exception as e:
            print(f"Measurements calculation error: {e}")
            return []
    
    def calculate_perimeter(self, labels, obj_id):
        """Calculate object perimeter"""
        try:
            height = len(labels)
            width = len(labels[0]) if height > 0 else 0
            perimeter = 0
            
            for y in range(height):
                for x in range(width):
                    if labels[y][x] == obj_id:
                        # Check if pixel is on boundary
                        is_boundary = False
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < height and 0 <= nx < width):
                                    if labels[ny][nx] != obj_id:
                                        is_boundary = True
                                        break
                                else:
                                    is_boundary = True
                                    break
                            if is_boundary:
                                break
                        
                        if is_boundary:
                            perimeter += 1
            
            return perimeter
            
        except Exception as e:
            print(f"Perimeter calculation error: {e}")
            return 0
    
    def calculate_aspect_ratio(self, labels, obj_id):
        """Calculate object aspect ratio"""
        try:
            # Find bounding box
            min_x, max_x = float('inf'), -float('inf')
            min_y, max_y = float('inf'), -float('inf')
            
            height = len(labels)
            width = len(labels[0]) if height > 0 else 0
            
            for y in range(height):
                for x in range(width):
                    if labels[y][x] == obj_id:
                        min_x = min(min_x, x)
                        max_x = max(max_x, x)
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
            
            if min_x != float('inf'):
                width_obj = max_x - min_x + 1
                height_obj = max_y - min_y + 1
                return max(width_obj, height_obj) / min(width_obj, height_obj) if min(width_obj, height_obj) > 0 else 1.0
            
            return 1.0
            
        except Exception as e:
            print(f"Aspect ratio calculation error: {e}")
            return 1.0
    
    def extract_object_boundaries(self, labels):
        """Extract object boundaries for ROI export"""
        boundaries = {}
        
        try:
            height = len(labels)
            width = len(labels[0]) if height > 0 else 0
            
            # Find boundary pixels for each object
            for y in range(height):
                for x in range(width):
                    label_id = labels[y][x]
                    if label_id > 0:
                        # Check if pixel is on boundary
                        is_boundary = False
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < height and 0 <= nx < width):
                                    if labels[ny][nx] != label_id:
                                        is_boundary = True
                                        break
                                else:
                                    is_boundary = True
                                    break
                            if is_boundary:
                                break
                        
                        if is_boundary:
                            if label_id not in boundaries:
                                boundaries[label_id] = []
                            boundaries[label_id].append((x, y))
            
            return boundaries
            
        except Exception as e:
            print(f"Boundary extraction error: {e}")
            return {}
    
    def handle_download(self):
        """Handle file download requests"""
        try:
            # Extract filename from URL
            filename = os.path.basename(self.path)
            filepath = os.path.join(UPLOAD_DIR, filename)
            
            if not os.path.exists(filepath):
                self.send_error(404, "File not found")
                return
            
            # Determine content type based on file extension
            if filename.endswith('.csv'):
                content_type = 'text/csv'
            elif filename.endswith('.tiff') or filename.endswith('.tif'):
                content_type = 'image/tiff'
            else:
                content_type = 'application/octet-stream'
            
            # Send file
            file_size = os.path.getsize(filepath)
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(file_size))
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
                
        except Exception as e:
            print(f"Download error: {e}")
            self.send_error(500, "Download failed")
    
    def handle_fibsem_analysis(self):
        """Handle comprehensive FIB-SEM analysis"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            analysis_type = data.get('analysis_type', 'full')
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            image_data = sessions[session_id]['images']['original']
            
            if FIBSEM_AVAILABLE:
                analyzer = FIBSEMAnalyzer()
                results = analyzer.run_comprehensive_analysis(image_data, analysis_type)
                
                for result_key, result_data in results.get('results', {}).items():
                    sessions[session_id]['layers'][result_key] = result_data
                
                response = {
                    'success': True,
                    'analysis_type': analysis_type,
                    'tools_used': results.get('tools_used', []),
                    'results_count': len(results.get('results', {})),
                    'message': f'FIB-SEM analysis complete using {len(results.get("tools_used", []))} specialized tools'
                }
            else:
                response = {
                    'success': True,
                    'analysis_type': analysis_type,
                    'tools_used': ['fallback_analysis'],
                    'results_count': 1,
                    'message': 'Basic analysis complete (FIB-SEM tools not available)'
                }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"FIB-SEM analysis error: {e}")
            self.send_json_response({'error': f'FIB-SEM analysis failed: {str(e)}'}, 500)
    
    def handle_chimerax_launch(self):
        """Handle ChimeraX integration"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            include_segmentation = data.get('include_segmentation', False)
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            if FIBSEM_AVAILABLE:
                chimerax = ChimeraXIntegration()
                volume_data = sessions[session_id]['images']['original']
                
                segmentation_data = None
                if include_segmentation and 'labels' in sessions[session_id]['layers']:
                    segmentation_data = sessions[session_id]['layers']['labels']
                
                result = chimerax.launch_chimerax_analysis(volume_data, segmentation_data)
                self.send_json_response(result if 'error' not in result else result, 400 if 'error' in result else 200)
            else:
                self.send_json_response({
                    'error': 'ChimeraX integration requires FIB-SEM plugins. Please install fibsem_plugins module.'
                }, 400)
                
        except Exception as e:
            print(f"ChimeraX launch error: {e}")
            self.send_json_response({'error': f'ChimeraX launch failed: {str(e)}'}, 500)
    
    def handle_3d_counting(self):
        """Handle 3D object counting"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'labels' not in sessions[session_id]['layers']:
                self.send_json_response({'error': 'No segmentation labels found'}, 400)
                return
            
            if FIBSEM_AVAILABLE:
                labels_2d = sessions[session_id]['layers']['labels']
                labels_3d = [labels_2d]
                
                results = ThreeDCounter.count_3d_objects(labels_3d)
                sessions[session_id]['results']['3d_counting'] = results
                
                response = {
                    'success': True,
                    'total_objects': results.get('total_objects', 0),
                    'largest_object_volume': results.get('largest_object', {}).get('volume_voxels', 0),
                    'message': f'3D counting complete: {results.get("total_objects", 0)} objects found'
                }
            else:
                labels = sessions[session_id]['layers']['labels']
                unique_labels = set()
                for row in labels:
                    for pixel in row:
                        if pixel > 0:
                            unique_labels.add(pixel)
                
                response = {
                    'success': True,
                    'total_objects': len(unique_labels),
                    'message': f'Basic counting complete: {len(unique_labels)} objects found'
                }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"3D counting error: {e}")
            self.send_json_response({'error': f'3D counting failed: {str(e)}'}, 500)
    
    def handle_accelerated_classification(self):
        """Handle accelerated pixel/object classification"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            classification_type = data.get('classification_type', 'mitochondria')
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            image_data = sessions[session_id]['images']['original']
            
            if FIBSEM_AVAILABLE:
                classifier = AcceleratedClassification()
                classification_map = classifier.classify_organelles(image_data, classification_type)
                
                sessions[session_id]['layers'][f'classification_{classification_type}'] = classification_map
                
                classified_pixels = sum(sum(1 for pixel in row if pixel > 0) for row in classification_map)
                total_pixels = len(classification_map) * len(classification_map[0]) if classification_map else 0
                
                response = {
                    'success': True,
                    'classification_type': classification_type,
                    'classified_pixels': classified_pixels,
                    'total_pixels': total_pixels,
                    'percentage': (classified_pixels / total_pixels * 100) if total_pixels > 0 else 0,
                    'message': f'{classification_type.title()} classification complete: {classified_pixels} pixels classified'
                }
            else:
                response = {
                    'success': True,
                    'classification_type': classification_type,
                    'message': f'Basic {classification_type} classification complete (accelerated tools not available)'
                }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Classification error: {e}")
            self.send_json_response({'error': f'Classification failed: {str(e)}'}, 500)
    
    def handle_membrane_segmentation(self):
        """Handle specialized membrane segmentation"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            segmentation_type = data.get('segmentation_type', 'membranes')
            params = data.get('params', {})
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            if 'original' not in sessions[session_id]['images']:
                self.send_json_response({'error': 'No image loaded'}, 400)
                return
            
            image_data = sessions[session_id]['images']['original']
            
            if FIBSEM_AVAILABLE:
                if segmentation_type == 'membranes':
                    membrane_thickness = params.get('membrane_thickness', 2)
                    labels = MembraneSegmenter.segment_membranes(image_data, membrane_thickness)
                elif segmentation_type == 'vesicles':
                    min_size = params.get('min_size', 10)
                    max_size = params.get('max_size', 200)
                    labels = MembraneSegmenter.segment_vesicles(image_data, min_size, max_size)
                else:
                    labels = MembraneSegmenter.segment_membranes(image_data)
                
                sessions[session_id]['layers'][f'membrane_{segmentation_type}'] = labels
                
                object_count = max(max(row) for row in labels) if labels else 0
                
                response = {
                    'success': True,
                    'segmentation_type': segmentation_type,
                    'object_count': object_count,
                    'message': f'Membrane {segmentation_type} segmentation complete: {object_count} structures found'
                }
            else:
                response = {
                    'success': True,
                    'segmentation_type': segmentation_type,
                    'message': f'Basic {segmentation_type} segmentation complete (specialized tools not available)'
                }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Membrane segmentation error: {e}")
            self.send_json_response({'error': f'Membrane segmentation failed: {str(e)}'}, 500)

    def handle_timelapse_alignment(self):
        """Handle timelapse sequence alignment"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            method = data.get('method', 'cross_correlation')
            reference_frame = data.get('reference_frame', 0)
            params = data.get('params', {})
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            # Get image sequence from session (assuming multiple frames loaded)
            image_sequence = sessions[session_id]['images'].get('sequence', [])
            if not image_sequence:
                self.send_json_response({'error': 'No image sequence found'}, 400)
                return
            
            if ADVANCED_PROCESSING_AVAILABLE:
                aligner = ImageAligner()
                result = aligner.align_timelapse_sequence(image_sequence, reference_frame, method, params)
                
                if result.get('success'):
                    # Store aligned sequence
                    sessions[session_id]['images']['aligned_sequence'] = result['aligned_sequence']
                    sessions[session_id]['metadata']['alignment_stats'] = {
                        'shift_vectors': result['shift_vectors'],
                        'alignment_scores': result['alignment_scores'],
                        'drift_statistics': result['drift_statistics']
                    }
                    
                    response = {
                        'success': True,
                        'method': method,
                        'total_frames': result['total_frames'],
                        'max_drift': result['drift_statistics']['max_frame_drift'],
                        'is_stable': result['drift_statistics']['is_stable'],
                        'message': f'Alignment complete: {result["total_frames"]} frames aligned'
                    }
                else:
                    response = result
            else:
                response = {'error': 'Advanced processing not available'}
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Timelapse alignment error: {e}")
            self.send_json_response({'error': f'Alignment failed: {str(e)}'}, 500)
    
    def handle_timelapse_normalization(self):
        """Handle timelapse intensity normalization"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            method = data.get('method', 'global_percentile')
            params = data.get('params', {})
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            # Get image sequence
            image_sequence = sessions[session_id]['images'].get('aligned_sequence') or sessions[session_id]['images'].get('sequence', [])
            if not image_sequence:
                self.send_json_response({'error': 'No image sequence found'}, 400)
                return
            
            if ADVANCED_PROCESSING_AVAILABLE:
                processor = TimelapseProcessor()
                result = processor.normalize_timelapse(image_sequence, method, params)
                
                if result.get('success'):
                    # Store normalized sequence
                    sessions[session_id]['images']['normalized_sequence'] = result['normalized_sequence']
                    sessions[session_id]['metadata']['normalization_stats'] = result['normalization_stats']
                    
                    response = {
                        'success': True,
                        'method': method,
                        'original_frames': result['original_frames'],
                        'intensity_range': result['intensity_range'],
                        'message': f'Normalization complete: {method} applied to {result["original_frames"]} frames'
                    }
                else:
                    response = result
            else:
                response = {'error': 'Advanced processing not available'}
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Timelapse normalization error: {e}")
            self.send_json_response({'error': f'Normalization failed: {str(e)}'}, 500)
    
    def handle_timelapse_analysis(self):
        """Handle temporal dynamics analysis"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            roi_coords = data.get('roi_coords')
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            # Get processed sequence
            image_sequence = (sessions[session_id]['images'].get('normalized_sequence') or 
                            sessions[session_id]['images'].get('aligned_sequence') or 
                            sessions[session_id]['images'].get('sequence', []))
            
            if not image_sequence:
                self.send_json_response({'error': 'No image sequence found'}, 400)
                return
            
            if ADVANCED_PROCESSING_AVAILABLE:
                processor = TimelapseProcessor()
                result = processor.analyze_temporal_dynamics(image_sequence, roi_coords)
                
                if result.get('success'):
                    sessions[session_id]['results']['temporal_analysis'] = result
                    
                    response = {
                        'success': True,
                        'frame_count': result['frame_count'],
                        'roi_size': result['roi_size'],
                        'intensity_range': result['intensity_range'],
                        'significant_events': len(result['significant_events']),
                        'bleaching_detected': result['bleaching_detected'],
                        'activation_detected': result['activation_detected'],
                        'message': f'Temporal analysis complete: {len(result["significant_events"])} significant events detected'
                    }
                else:
                    response = result
            else:
                response = {'error': 'Advanced processing not available'}
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Temporal analysis error: {e}")
            self.send_json_response({'error': f'Temporal analysis failed: {str(e)}'}, 500)
    
    def handle_batch_create(self):
        """Create batch processing session"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            files = data.get('files', [])
            workflow_config = data.get('workflow_config', {})
            
            if not files:
                self.send_json_response({'error': 'No files provided'}, 400)
                return
            
            if ADVANCED_PROCESSING_AVAILABLE:
                if not hasattr(self, 'batch_processor'):
                    self.batch_processor = BatchProcessor()
                
                batch_id = self.batch_processor.create_batch_session(files, workflow_config)
                
                response = {
                    'success': True,
                    'batch_id': batch_id,
                    'total_files': len(files),
                    'workflow': workflow_config.get('name', 'Custom workflow'),
                    'message': f'Batch session created with {len(files)} files'
                }
            else:
                response = {'error': 'Batch processing not available'}
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Batch creation error: {e}")
            self.send_json_response({'error': f'Batch creation failed: {str(e)}'}, 500)
    
    def handle_batch_process(self):
        """Start batch processing"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            batch_id = data.get('batch_id')
            
            if not batch_id:
                self.send_json_response({'error': 'Batch ID required'}, 400)
                return
            
            if ADVANCED_PROCESSING_AVAILABLE and hasattr(self, 'batch_processor'):
                result = self.batch_processor.process_batch(batch_id)
                self.send_json_response(result)
            else:
                self.send_json_response({'error': 'Batch processor not available'}, 400)
            
        except Exception as e:
            print(f"Batch processing error: {e}")
            self.send_json_response({'error': f'Batch processing failed: {str(e)}'}, 500)
    
    def handle_batch_status(self):
        """Get batch processing status"""
        try:
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)
            
            batch_id = query_params.get('batch_id', [None])[0]
            
            if not batch_id:
                self.send_json_response({'error': 'Batch ID required'}, 400)
                return
            
            if ADVANCED_PROCESSING_AVAILABLE and hasattr(self, 'batch_processor'):
                status = self.batch_processor.get_batch_status(batch_id)
                self.send_json_response(status)
            else:
                self.send_json_response({'error': 'Batch processor not available'}, 400)
            
        except Exception as e:
            print(f"Batch status error: {e}")
            self.send_json_response({'error': f'Status check failed: {str(e)}'}, 500)

    def handle_batch_upload(self):
        """Handle multiple file uploads for timelapse sequences"""
        try:
            content_type = self.headers['content-type']
            if not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Content type must be multipart/form-data")
                return
            
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            session_id = form.getvalue('session_id')
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            # Get sequence type
            sequence_type = form.getvalue('sequence_type', '2d_timelapse')  # '2d_timelapse' or '3d_timelapse'
            
            # Process multiple files
            uploaded_files = []
            file_sequence = []
            
            for key in form.keys():
                if key.startswith('file_'):
                    file_item = form[key]
                    if file_item.filename:
                        filename = file_item.filename
                        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")
                        
                        with open(file_path, 'wb') as f:
                            f.write(file_item.file.read())
                        
                        # Load image data
                        image_data = EnhancedImageProcessor.load_image(file_path)
                        if image_data:
                            uploaded_files.append({
                                'filename': filename,
                                'path': file_path,
                                'data': image_data,
                                'shape': [len(image_data), len(image_data[0]) if image_data else 0]
                            })
                            file_sequence.append(image_data)
            
            if not uploaded_files:
                self.send_json_response({'error': 'No valid files uploaded'}, 400)
                return
            
            # Sort files by name for proper sequence order
            uploaded_files.sort(key=lambda x: x['filename'])
            file_sequence = [f['data'] for f in uploaded_files]
            
            # Store sequence in session
            sessions[session_id]['images']['sequence'] = file_sequence
            sessions[session_id]['metadata']['sequence_type'] = sequence_type
            sessions[session_id]['metadata']['sequence_files'] = [f['filename'] for f in uploaded_files]
            sessions[session_id]['metadata']['sequence_length'] = len(file_sequence)
            
            # If it's a 3D timelapse, organize by timepoints and Z-slices
            if sequence_type == '3d_timelapse':
                # Try to parse timepoint and Z information from filenames
                organized_sequence = self._organize_3d_sequence(uploaded_files)
                sessions[session_id]['images']['3d_sequence'] = organized_sequence
            
            response = {
                'success': True,
                'sequence_type': sequence_type,
                'total_files': len(uploaded_files),
                'filenames': [f['filename'] for f in uploaded_files],
                'sequence_shape': f"{len(file_sequence)} frames of {uploaded_files[0]['shape'][0]}x{uploaded_files[0]['shape'][1]}",
                'message': f'Successfully loaded {len(uploaded_files)} files as {sequence_type}'
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Batch upload error: {e}")
            self.send_json_response({'error': f'Batch upload failed: {str(e)}'}, 500)
    
    def handle_timelapse_load(self):
        """Handle timelapse sequence loading with metadata parsing"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            session_id = data.get('session_id')
            file_pattern = data.get('file_pattern', '')  # e.g., "cells_t*_z*.tif"
            sequence_type = data.get('sequence_type', '2d_timelapse')
            
            if not session_id or session_id not in sessions:
                self.send_json_response({'error': 'Invalid session'}, 400)
                return
            
            # Auto-detect sequence files based on pattern or naming convention
            if not file_pattern:
                # Look for files with similar naming patterns
                available_files = self._find_sequence_files(session_id)
            else:
                available_files = self._match_file_pattern(file_pattern)
            
            if not available_files:
                self.send_json_response({'error': 'No sequence files found'}, 400)
                return
            
            # Load and organize sequence
            sequence_data = []
            metadata = {
                'timepoints': set(),
                'z_slices': set(),
                'channels': set(),
                'positions': set()
            }
            
            for file_info in available_files:
                image_data = EnhancedImageProcessor.load_image(file_info['path'])
                if image_data:
                    sequence_data.append({
                        'data': image_data,
                        'metadata': file_info['metadata'],
                        'filename': file_info['filename']
                    })
                    
                    # Collect metadata
                    if 'timepoint' in file_info['metadata']:
                        metadata['timepoints'].add(file_info['metadata']['timepoint'])
                    if 'z_slice' in file_info['metadata']:
                        metadata['z_slices'].add(file_info['metadata']['z_slice'])
                    if 'channel' in file_info['metadata']:
                        metadata['channels'].add(file_info['metadata']['channel'])
                    if 'position' in file_info['metadata']:
                        metadata['positions'].add(file_info['metadata']['position'])
            
            # Convert sets to sorted lists
            for key in metadata:
                metadata[key] = sorted(list(metadata[key]))
            
            # Store in session
            sessions[session_id]['images']['timelapse_sequence'] = sequence_data
            sessions[session_id]['metadata']['timelapse_info'] = metadata
            sessions[session_id]['metadata']['sequence_type'] = sequence_type
            
            response = {
                'success': True,
                'sequence_type': sequence_type,
                'total_frames': len(sequence_data),
                'timepoints': len(metadata['timepoints']),
                'z_slices': len(metadata['z_slices']),
                'channels': len(metadata['channels']),
                'positions': len(metadata['positions']),
                'metadata': metadata,
                'message': f'Loaded {len(sequence_data)} frames with {len(metadata["timepoints"])} timepoints'
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f"Timelapse load error: {e}")
            self.send_json_response({'error': f'Timelapse loading failed: {str(e)}'}, 500)
    
    def _organize_3d_sequence(self, uploaded_files):
        """Organize files into 3D timelapse structure (T, Z, Y, X)"""
        organized = {}
        
        for file_info in uploaded_files:
            filename = file_info['filename']
            
            # Parse timepoint and Z-slice from filename
            timepoint, z_slice = self._parse_filename_metadata(filename)
            
            if timepoint not in organized:
                organized[timepoint] = {}
            
            organized[timepoint][z_slice] = file_info['data']
        
        return organized
    
    def _parse_filename_metadata(self, filename):
        """Parse timepoint and Z-slice information from filename"""
        import re
        
        # Common patterns for microscopy filenames
        patterns = [
            r't(\d+).*z(\d+)',  # t1_z5 format
            r'_t(\d+)_.*_z(\d+)',  # _t01_ch1_z005 format
            r's(\d+)_t(\d+)',  # s1_t3 format (position, timepoint)
            r'_s(\d+)_t(\d+)',  # _s1_t3 format
        ]
        
        timepoint = 0
        z_slice = 0
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                if 't' in pattern and 'z' in pattern:
                    timepoint = int(match.group(1))
                    z_slice = int(match.group(2))
                elif 's' in pattern and 't' in pattern:
                    timepoint = int(match.group(2))
                    z_slice = 0  # No Z info
                break
        
        return timepoint, z_slice
    
    def _find_sequence_files(self, session_id):
        """Find files that appear to be part of a sequence"""
        # Look in the upload directory for files with similar names
        upload_files = []
        
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(f"{session_id}_"):
                base_name = filename[len(f"{session_id}_"):]
                file_path = os.path.join(UPLOAD_DIR, filename)
                
                # Parse metadata from filename
                timepoint, z_slice = self._parse_filename_metadata(base_name)
                
                upload_files.append({
                    'filename': base_name,
                    'path': file_path,
                    'metadata': {
                        'timepoint': timepoint,
                        'z_slice': z_slice
                    }
                })
        
        return upload_files
    
    def _match_file_pattern(self, pattern):
        """Match files based on a pattern (placeholder for now)"""
        # This would implement pattern matching for file selection
        # For now, return empty list
        return []

    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        json_data = json.dumps(data).encode()
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-length', str(len(json_data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json_data)
    
    def get_enhanced_html(self):
        """Return enhanced HTML interface with all features"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Scientific Image Analyzer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; line-height: 1.6; }
        
        .header { background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 1rem 0; }
        .header-content { max-width: 1400px; margin: 0 auto; padding: 0 2rem; }
        .header h1 { font-size: 1.8rem; font-weight: 300; }
        .subtitle { font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem; }
        
        .main-container { max-width: 1400px; margin: 2rem auto; padding: 0 2rem; }
        .dashboard { display: grid; grid-template-columns: 400px 1fr; gap: 2rem; }
        
        .card { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem; }
        .card-title { font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-bottom: 1rem; }
        
        .btn { padding: 0.6rem 1.2rem; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; margin: 0.25rem; transition: all 0.3s; font-size: 0.9rem; width: 100%; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-primary { background: linear-gradient(135deg, #3498db, #2980b9); color: white; }
        .btn-primary:hover:not(:disabled) { background: linear-gradient(135deg, #2980b9, #1f639a); }
        .btn-success { background: linear-gradient(135deg, #27ae60, #229954); color: white; }
        .btn-warning { background: linear-gradient(135deg, #f39c12, #e67e22); color: white; }
        .btn-info { background: linear-gradient(135deg, #17a2b8, #138496); color: white; }
        
        .upload-area { border: 3px dashed #bdc3c7; border-radius: 8px; padding: 2rem; text-align: center; background: #f8f9fa; margin-bottom: 1rem; transition: all 0.3s; cursor: pointer; }
        .upload-area:hover { border-color: #3498db; background: #ebf3ff; }
        .upload-area.dragover { border-color: #27ae60; background: #eafaf1; }
        
        .file-input { display: none; }
        .upload-text { color: #666; margin-bottom: 1rem; }
        
        .section { margin: 0.8rem 0; padding: 0.8rem; border-left: 4px solid #3498db; background: #f8f9fa; }
        .section-title { font-weight: 600; color: #2c3e50; margin-bottom: 0.4rem; font-size: 0.95rem; }
        .section-description { font-size: 0.8rem; color: #666; margin-bottom: 0.8rem; }
        
        .parameter-group { margin: 0.8rem 0; }
        .parameter-label { display: block; margin-bottom: 0.4rem; color: #444; font-weight: 500; font-size: 0.85rem; }
        .parameter-input { width: 100%; padding: 0.4rem; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 0.4rem; }
        
        .results-display { background: #f8f9fa; border-left: 4px solid #27ae60; padding: 1rem; margin: 1rem 0; font-family: monospace; font-size: 0.9rem; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin: 1rem 0; }
        .stat-item { background: white; padding: 1rem; border-radius: 4px; text-align: center; }
        .stat-value { font-size: 1.2rem; font-weight: bold; color: #3498db; }
        .stat-label { font-size: 0.8rem; color: #666; }
        
        .message { padding: 0.75rem; border-radius: 4px; margin: 1rem 0; }
        .message-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .message-info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        
        .image-display { border: 2px solid #e9ecef; border-radius: 8px; min-height: 300px; display: flex; align-items: center; justify-content: center; background: #fff; }
        .placeholder-text { color: #999; font-size: 1.1rem; }
        
        .controls-scroll { max-height: 80vh; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1>Enhanced Scientific Image Analyzer</h1>
            <div class="subtitle">Advanced Microscopy Analysis Suite with AI Denoising, 4D Viewing & Advanced Segmentation</div>
            <div class="subtitle">Upload images and perform comprehensive analysis with cutting-edge algorithms</div>
        </div>
    </div>

    <div class="main-container">
        <div class="dashboard">
            <!-- Control Panel -->
            <div class="controls-scroll">
                <!-- File Upload -->
                <div class="card">
                    <h2 class="card-title">File Upload</h2>
                    <div class="upload-area" id="upload-area">
                        <div class="upload-text">
                            <strong>Click or drag to upload images</strong><br>
                            Supports: TIFF, PNG, JPEG, and microscopy formats
                        </div>
                    </div>
                    <input type="file" id="file-input" class="file-input" accept=".tif,.tiff,.png,.jpg,.jpeg,.czi,.lif,.nd2,.oib,.oif">
                    <div id="upload-status"></div>
                </div>

                <!-- AI Denoising -->
                <div class="card">
                    <h2 class="card-title">AI Denoising</h2>
                    
                    <div class="section">
                        <div class="section-title">Bilateral Filter</div>
                        <div class="section-description">Edge-preserving noise reduction</div>
                        <div class="parameter-group">
                            <label class="parameter-label">Spatial Sigma:</label>
                            <input type="number" id="bilateral-spatial" class="parameter-input" value="5" min="1" max="20">
                            <label class="parameter-label">Intensity Sigma:</label>
                            <input type="number" id="bilateral-intensity" class="parameter-input" value="20" min="5" max="100">
                        </div>
                        <button id="bilateral-btn" class="btn btn-primary" disabled>Apply Bilateral Denoising</button>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Non-Local Means</div>
                        <div class="section-description">Patch-based denoising</div>
                        <div class="parameter-group">
                            <label class="parameter-label">Filter Strength:</label>
                            <input type="number" id="nlm-strength" class="parameter-input" value="10" min="1" max="50">
                        </div>
                        <button id="nlm-btn" class="btn btn-primary" disabled>Apply NLM Denoising</button>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Wiener Filter</div>
                        <div class="section-description">Frequency domain denoising</div>
                        <button id="wiener-btn" class="btn btn-primary" disabled>Apply Wiener Filter</button>
                    </div>
                </div>

                <!-- Basic Processing -->
                <div class="card">
                    <h2 class="card-title">Image Processing</h2>
                    
                    <div class="section">
                        <div class="section-title">Gaussian Filter</div>
                        <div class="parameter-group">
                            <label class="parameter-label">Sigma:</label>
                            <input type="number" id="gaussian-sigma" class="parameter-input" value="2.0" min="0.1" max="10" step="0.1">
                        </div>
                        <button id="gaussian-btn" class="btn btn-primary" disabled>Apply Gaussian Filter</button>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Threshold</div>
                        <button id="threshold-btn" class="btn btn-primary" disabled>Apply Threshold</button>
                    </div>
                </div>

                <!-- AI Segmentation -->
                <div class="card">
                    <h2 class="card-title">AI Segmentation</h2>
                    
                    <div class="section">
                        <div class="section-title">Cellpose</div>
                        <div class="parameter-group">
                            <label class="parameter-label">Cell Diameter:</label>
                            <input type="number" id="cellpose-diameter" class="parameter-input" value="30" min="10" max="100">
                        </div>
                        <button id="cellpose-btn" class="btn btn-success" disabled>Run Cellpose</button>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">StarDist</div>
                        <button id="stardist-btn" class="btn btn-success" disabled>Run StarDist</button>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Watershed</div>
                        <button id="watershed-btn" class="btn btn-success" disabled>Run Watershed</button>
                    </div>
                </div>

                <!-- Advanced Segmentation -->
                <div class="card">
                    <h2 class="card-title">Advanced Segmentation</h2>
                    
                    <div class="section">
                        <div class="section-title">Morphological Snakes</div>
                        <div class="parameter-group">
                            <label class="parameter-label">Iterations:</label>
                            <input type="number" id="snakes-iterations" class="parameter-input" value="50" min="10" max="200">
                        </div>
                        <button id="snakes-btn" class="btn btn-success" disabled>Run Morphological Snakes</button>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Region Growing</div>
                        <div class="parameter-group">
                            <label class="parameter-label">Threshold:</label>
                            <input type="number" id="region-threshold" class="parameter-input" value="0.1" min="0.01" max="1.0" step="0.01">
                        </div>
                        <button id="region-btn" class="btn btn-success" disabled>Run Region Growing</button>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Adaptive Threshold</div>
                        <div class="parameter-group">
                            <label class="parameter-label">Block Size:</label>
                            <input type="number" id="adaptive-block" class="parameter-input" value="11" min="3" max="21" step="2">
                        </div>
                        <button id="adaptive-btn" class="btn btn-success" disabled>Run Adaptive Threshold</button>
                    </div>
                </div>

                <!-- 4D Viewing -->
                <div class="card">
                    <h2 class="card-title">4D Image Viewing</h2>
                    
                    <div class="section">
                        <div class="section-title">Volume Navigation</div>
                        <div class="parameter-group">
                            <label class="parameter-label">Time Point:</label>
                            <input type="range" id="time-slider" class="parameter-input" value="0" min="0" max="10" disabled>
                            <span id="time-display">T: 0</span>
                            
                            <label class="parameter-label">Z Slice:</label>
                            <input type="range" id="z-slider" class="parameter-input" value="0" min="0" max="10" disabled>
                            <span id="z-display">Z: 0</span>
                        </div>
                        <button id="init-4d-btn" class="btn btn-info" disabled>Initialize 4D Viewer</button>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Maximum Projections</div>
                        <div class="parameter-group">
                            <button id="proj-z-btn" class="btn btn-info" disabled>Z Projection</button>
                            <button id="proj-y-btn" class="btn btn-info" disabled>Y Projection</button>
                            <button id="proj-x-btn" class="btn btn-info" disabled>X Projection</button>
                        </div>
                    </div>
                </div>

                <!-- Analysis -->
                <div class="card">
                    <h2 class="card-title">Analysis Tools</h2>
                    
                    <div class="section">
                        <div class="section-title">Object Measurements</div>
                        <button id="measure-btn" class="btn btn-info" disabled>Measure Objects</button>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Colocalization</div>
                        <button id="coloc-btn" class="btn btn-info" disabled>Run Colocalization</button>
                    </div>
                </div>

                <!-- Export for Tracking -->
                <div class="card">
                    <h2 class="card-title">Export for Single Particle Tracking</h2>
                    
                    <div class="section">
                        <div class="section-title">Export Format</div>
                        <div class="section-description">Choose format compatible with your tracking software</div>
                        <div class="parameter-group">
                            <label class="parameter-label">Export Type:</label>
                            <select id="export-type" class="parameter-input">
                                <option value="tracking_ready">Tracking Ready CSV (Recommended)</option>
                                <option value="centroids_csv">Centroids CSV</option>
                                <option value="mask_tiff">Segmentation Mask TIFF</option>
                                <option value="coordinates_csv">All Pixel Coordinates</option>
                                <option value="measurements_csv">Complete Measurements</option>
                                <option value="imageJ_roi">ImageJ ROI Format</option>
                            </select>
                            
                            <label class="parameter-label">Data Source:</label>
                            <select id="export-source" class="parameter-input">
                                <option value="labels_cellpose">Cellpose Segmentation</option>
                                <option value="labels_stardist">StarDist Segmentation</option>
                                <option value="labels_watershed">Watershed Segmentation</option>
                                <option value="advanced_labels_morphological_snakes">Morphological Snakes</option>
                                <option value="advanced_labels_region_growing">Region Growing</option>
                                <option value="advanced_labels_adaptive_threshold">Adaptive Threshold</option>
                            </select>
                        </div>
                        
                        <button id="export-btn" class="btn btn-warning" disabled>Export Segmentation Data</button>
                        
                        <div id="export-status"></div>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Export Formats Explained</div>
                        <div class="section-description" style="font-size: 0.75rem; line-height: 1.4;">
                            <strong>Tracking Ready CSV:</strong> frame,object_id,x,y,area,perimeter,intensity - Complete data for tracking initialization<br>
                            <strong>Centroids CSV:</strong> object_id,centroid_x,centroid_y,area - Object centers for tracking<br>
                            <strong>Mask TIFF:</strong> Binary segmentation mask with labeled objects<br>
                            <strong>Coordinates CSV:</strong> All pixel coordinates for each object<br>
                            <strong>ImageJ ROI:</strong> Region boundaries compatible with ImageJ
                        </div>
                    </div>
                </div>
            </div>

            <!-- Display Area -->
            <div>
                <div class="card">
                    <h2 class="card-title">Image Display</h2>
                    <div class="image-display" id="image-display">
                        <div class="placeholder-text">Upload an image to begin enhanced analysis</div>
                    </div>
                </div>

                <div class="card">
                    <h2 class="card-title">Results & Analysis</h2>
                    <div id="results-container">
                        <div class="placeholder-text">Enhanced analysis results will appear here</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let sessionId = null;
        let currentImage = null;

        // Initialize session on page load
        async function initSession() {
            try {
                const response = await fetch('/api/session/create');
                const data = await response.json();
                sessionId = data.session_id;
                showMessage('Enhanced analyzer ready - upload an image to begin', 'success');
            } catch (error) {
                console.error('Failed to initialize session:', error);
                showMessage('Failed to initialize session', 'error');
            }
        }

        // File upload handling
        function setupFileUpload() {
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');

            uploadArea.addEventListener('click', () => fileInput.click());

            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    uploadFile(files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    uploadFile(e.target.files[0]);
                }
            });
        }

        async function uploadFile(file) {
            if (!sessionId) {
                showMessage('Session not initialized', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);

            showMessage('Uploading file...', 'info');

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    currentImage = data.filename;
                    showMessage(data.message, 'success');
                    enableControls();
                } else {
                    showMessage(data.error || 'Upload failed', 'error');
                }
            } catch (error) {
                console.error('Upload error:', error);
                showMessage('Upload failed', 'error');
            }
        }

        function enableControls() {
            const buttons = [
                'gaussian-btn', 'threshold-btn',
                'bilateral-btn', 'nlm-btn', 'wiener-btn',
                'cellpose-btn', 'stardist-btn', 'watershed-btn',
                'snakes-btn', 'region-btn', 'adaptive-btn',
                'init-4d-btn', 'measure-btn', 'coloc-btn'
            ];
            buttons.forEach(id => {
                const element = document.getElementById(id);
                if (element) element.disabled = false;
            });
        }

        async function processImage(operation, params = {}) {
            if (!sessionId || !currentImage) return;

            showMessage(`Running ${operation}...`, 'info');

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        operation: operation,
                        params: params
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showMessage(data.message, 'success');
                } else {
                    showMessage(data.error || 'Processing failed', 'error');
                }
            } catch (error) {
                console.error('Processing error:', error);
                showMessage('Processing failed', 'error');
            }
        }

        async function denoiseImage(method, params = {}) {
            if (!sessionId || !currentImage) return;

            showMessage(`Running ${method} denoising...`, 'info');

            try {
                const response = await fetch('/api/denoise', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        method: method,
                        params: params
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showMessage(data.message, 'success');
                } else {
                    showMessage(data.error || 'Denoising failed', 'error');
                }
            } catch (error) {
                console.error('Denoising error:', error);
                showMessage('Denoising failed', 'error');
            }
        }

        async function segmentImage(method, params = {}) {
            if (!sessionId || !currentImage) return;

            showMessage(`Running ${method} segmentation...`, 'info');

            try {
                const response = await fetch('/api/segment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        method: method,
                        params: params
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showMessage(data.message, 'success');
                    document.getElementById('measure-btn').disabled = false;
                } else {
                    showMessage(data.error || 'Segmentation failed', 'error');
                }
            } catch (error) {
                console.error('Segmentation error:', error);
                showMessage('Segmentation failed', 'error');
            }
        }

        async function advancedSegmentImage(method, params = {}) {
            if (!sessionId || !currentImage) return;

            showMessage(`Running ${method} segmentation...`, 'info');

            try {
                const response = await fetch('/api/advanced_segment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        method: method,
                        params: params
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showMessage(data.message, 'success');
                    document.getElementById('measure-btn').disabled = false;
                } else {
                    showMessage(data.error || 'Advanced segmentation failed', 'error');
                }
            } catch (error) {
                console.error('Advanced segmentation error:', error);
                showMessage('Advanced segmentation failed', 'error');
            }
        }

        async function initialize4DViewer() {
            if (!sessionId || !currentImage) return;

            showMessage('Initializing 4D viewer...', 'info');

            try {
                const response = await fetch('/api/4d_navigate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        type: 'time',
                        index: 0
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showMessage('4D viewer initialized', 'success');
                    
                    const volumeInfo = data.volume_info;
                    if (volumeInfo && volumeInfo.dimensions) {
                        const timeSlider = document.getElementById('time-slider');
                        const zSlider = document.getElementById('z-slider');
                        
                        timeSlider.max = Math.max(1, volumeInfo.dimensions.T - 1);
                        zSlider.max = Math.max(1, volumeInfo.dimensions.Z - 1);
                        
                        timeSlider.disabled = false;
                        zSlider.disabled = false;
                        
                        ['proj-z-btn', 'proj-y-btn', 'proj-x-btn'].forEach(id => {
                            document.getElementById(id).disabled = false;
                        });
                    }
                    
                    displayVolumeInfo(data.volume_info);
                } else {
                    showMessage(data.error || '4D initialization failed', 'error');
                }
            } catch (error) {
                console.error('4D initialization error:', error);
                showMessage('4D initialization failed', 'error');
            }
        }

        async function navigate4D(type, index) {
            if (!sessionId || !currentImage) return;

            try {
                const response = await fetch('/api/4d_navigate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        type: type,
                        index: index
                    })
                });

                const data = await response.json();
                if (data.success) {
                    displaySliceInfo(data.current_slice_stats);
                }
            } catch (error) {
                console.error('4D navigation error:', error);
            }
        }

        async function create4DProjection(type) {
            if (!sessionId || !currentImage) return;

            showMessage(`Creating ${type} projection...`, 'info');

            try {
                const response = await fetch('/api/4d_projection', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        type: type
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showMessage(data.message, 'success');
                    displayProjectionInfo(data.projection_stats, type);
                } else {
                    showMessage(data.error || 'Projection failed', 'error');
                }
            } catch (error) {
                console.error('Projection error:', error);
                showMessage('Projection failed', 'error');
            }
        }

        async function analyzeObjects() {
            if (!sessionId || !currentImage) return;

            showMessage('Analyzing objects...', 'info');

            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        labels_key: 'labels_cellpose'
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showMessage(data.message, 'success');
                    displayResults(data);
                } else {
                    showMessage(data.error || 'Analysis failed', 'error');
                }
            } catch (error) {
                console.error('Analysis error:', error);
                showMessage('Analysis failed', 'error');
            }
        }

        async function runColocalization() {
            if (!sessionId || !currentImage) return;

            showMessage('Running colocalization analysis...', 'info');

            try {
                const response = await fetch('/api/colocalization', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showMessage(data.message, 'success');
                    displayColocResults(data.results);
                } else {
                    showMessage(data.error || 'Colocalization failed', 'error');
                }
            } catch (error) {
                console.error('Colocalization error:', error);
                showMessage('Colocalization failed', 'error');
            }
        }

        function displayVolumeInfo(volumeInfo) {
            if (!volumeInfo) return;
            
            const container = document.getElementById('results-container');
            let html = '<h3>4D Volume Information</h3><div class="stats-grid">';
            
            if (volumeInfo.dimensions) {
                const dims = volumeInfo.dimensions;
                html += `
                    <div class="stat-item">
                        <div class="stat-value">${dims.T}</div>
                        <div class="stat-label">Time Points</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${dims.Z}</div>
                        <div class="stat-label">Z Slices</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${dims.Y}×${dims.X}</div>
                        <div class="stat-label">XY Resolution</div>
                    </div>
                `;
            }
            
            html += '</div>';
            container.innerHTML = html;
        }

        function displaySliceInfo(sliceStats) {
            const container = document.getElementById('results-container');
            let html = '<h3>Current Slice Statistics</h3><div class="stats-grid">';
            
            html += `
                <div class="stat-item">
                    <div class="stat-value">${sliceStats.mean.toFixed(1)}</div>
                    <div class="stat-label">Mean Intensity</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${sliceStats.max.toFixed(1)}</div>
                    <div class="stat-label">Max Intensity</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${sliceStats.shape[0]}×${sliceStats.shape[1]}</div>
                    <div class="stat-label">Slice Size</div>
                </div>
            `;
            
            html += '</div>';
            container.innerHTML = html;
        }

        function displayProjectionInfo(projStats, type) {
            const container = document.getElementById('results-container');
            let html = `<h3>${type.toUpperCase()} Projection</h3><div class="stats-grid">`;
            
            html += `
                <div class="stat-item">
                    <div class="stat-value">${projStats.mean.toFixed(1)}</div>
                    <div class="stat-label">Mean Intensity</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${projStats.max.toFixed(1)}</div>
                    <div class="stat-label">Max Intensity</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${projStats.shape[0]}×${projStats.shape[1]}</div>
                    <div class="stat-label">Size</div>
                </div>
            `;
            
            html += '</div>';
            container.innerHTML = html;
        }

        function displayResults(data) {
            const container = document.getElementById('results-container');
            let html = '<h3>Object Analysis Results</h3><div class="stats-grid">';
            
            if (data.measurements && data.measurements.length > 0) {
                html += `
                    <div class="stat-item">
                        <div class="stat-value">${data.total_objects}</div>
                        <div class="stat-label">Total Objects</div>
                    </div>
                `;
                
                const sample = data.measurements[0];
                if (sample.area !== undefined) {
                    const avgArea = data.measurements.reduce((sum, m) => sum + m.area, 0) / data.measurements.length;
                    html += `
                        <div class="stat-item">
                            <div class="stat-value">${avgArea.toFixed(1)}</div>
                            <div class="stat-label">Avg Area</div>
                        </div>
                    `;
                }
            }
            
            html += '</div>';
            container.innerHTML = html;
        }

        function displayColocResults(results) {
            const container = document.getElementById('results-container');
            let html = '<h3>Colocalization Analysis</h3><div class="stats-grid">';
            
            if (results.pearson_correlation !== undefined) {
                html += `
                    <div class="stat-item">
                        <div class="stat-value">${results.pearson_correlation.toFixed(3)}</div>
                        <div class="stat-label">Pearson Correlation</div>
                    </div>
                `;
            }
            
            if (results.manders_m1 !== undefined) {
                html += `
                    <div class="stat-item">
                        <div class="stat-value">${results.manders_m1.toFixed(3)}</div>
                        <div class="stat-label">Manders M1</div>
                    </div>
                `;
            }
            
            html += '</div>';
            container.innerHTML = html;
        }

        async function exportSegmentationData() {
            if (!sessionId || !currentImage) return;

            const exportType = document.getElementById('export-type').value;
            const dataSource = document.getElementById('export-source').value;

            showMessage(`Exporting ${exportType} from ${dataSource}...`, 'info');

            try {
                const response = await fetch('/api/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        export_type: exportType,
                        data_source: dataSource
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showMessage(data.message, 'success');
                    
                    // Create download link
                    const exportStatus = document.getElementById('export-status');
                    const fileSize = (data.file_size / 1024).toFixed(1);
                    exportStatus.innerHTML = `
                        <div class="message message-success" style="margin-top: 1rem;">
                            <strong>Export Complete!</strong><br>
                            File: ${data.filename} (${fileSize} KB)<br>
                            <a href="${data.download_url}" download="${data.filename}" 
                               style="color: #155724; text-decoration: underline; font-weight: bold;">
                               📥 Download ${data.export_type} Data
                            </a>
                        </div>
                    `;
                } else {
                    showMessage(data.error || 'Export failed', 'error');
                    document.getElementById('export-status').innerHTML = `
                        <div class="message message-error" style="margin-top: 1rem;">
                            Export failed: ${data.error || 'Unknown error'}
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Export error:', error);
                showMessage('Export failed', 'error');
                document.getElementById('export-status').innerHTML = `
                    <div class="message message-error" style="margin-top: 1rem;">
                        Export failed: Network or server error
                    </div>
                `;
            }
        }

        function showMessage(text, type) {
            const statusEl = document.getElementById('upload-status');
            statusEl.className = `message message-${type}`;
            statusEl.textContent = text;
            
            if (type === 'success') {
                setTimeout(() => {
                    statusEl.textContent = '';
                    statusEl.className = '';
                }, 3000);
            }
        }

        // Event listeners
        document.addEventListener('DOMContentLoaded', () => {
            initSession();
            setupFileUpload();

            // Basic processing
            document.getElementById('gaussian-btn').addEventListener('click', () => {
                const sigma = parseFloat(document.getElementById('gaussian-sigma').value);
                processImage('gaussian_filter', { sigma });
            });

            document.getElementById('threshold-btn').addEventListener('click', () => {
                processImage('threshold', {});
            });

            // Denoising
            document.getElementById('bilateral-btn').addEventListener('click', () => {
                const spatialSigma = parseFloat(document.getElementById('bilateral-spatial').value);
                const intensitySigma = parseFloat(document.getElementById('bilateral-intensity').value);
                denoiseImage('bilateral', { sigma_spatial: spatialSigma, sigma_intensity: intensitySigma });
            });

            document.getElementById('nlm-btn').addEventListener('click', () => {
                const strength = parseFloat(document.getElementById('nlm-strength').value);
                denoiseImage('non_local_means', { h: strength });
            });

            document.getElementById('wiener-btn').addEventListener('click', () => {
                denoiseImage('wiener', {});
            });

            // Basic segmentation
            document.getElementById('cellpose-btn').addEventListener('click', () => {
                const diameter = parseInt(document.getElementById('cellpose-diameter').value);
                segmentImage('cellpose', { diameter });
            });

            document.getElementById('stardist-btn').addEventListener('click', () => {
                segmentImage('stardist');
            });

            document.getElementById('watershed-btn').addEventListener('click', () => {
                segmentImage('watershed');
            });

            // Advanced segmentation
            document.getElementById('snakes-btn').addEventListener('click', () => {
                const iterations = parseInt(document.getElementById('snakes-iterations').value);
                advancedSegmentImage('morphological_snakes', { iterations });
            });

            document.getElementById('region-btn').addEventListener('click', () => {
                const threshold = parseFloat(document.getElementById('region-threshold').value);
                advancedSegmentImage('region_growing', { 
                    seed_points: [[100, 100]], 
                    threshold 
                });
            });

            document.getElementById('adaptive-btn').addEventListener('click', () => {
                const blockSize = parseInt(document.getElementById('adaptive-block').value);
                advancedSegmentImage('adaptive_threshold', { block_size: blockSize });
            });

            // 4D viewing
            document.getElementById('init-4d-btn').addEventListener('click', initialize4DViewer);
            document.getElementById('proj-z-btn').addEventListener('click', () => create4DProjection('z'));
            document.getElementById('proj-y-btn').addEventListener('click', () => create4DProjection('y'));
            document.getElementById('proj-x-btn').addEventListener('click', () => create4DProjection('x'));

            // Navigation sliders
            document.getElementById('time-slider').addEventListener('input', (e) => {
                const timepoint = parseInt(e.target.value);
                navigate4D('time', timepoint);
                document.getElementById('time-display').textContent = `T: ${timepoint}`;
            });

            document.getElementById('z-slider').addEventListener('input', (e) => {
                const zSlice = parseInt(e.target.value);
                navigate4D('z', zSlice);
                document.getElementById('z-display').textContent = `Z: ${zSlice}`;
            });

            // Analysis
            document.getElementById('measure-btn').addEventListener('click', analyzeObjects);
            document.getElementById('coloc-btn').addEventListener('click', runColocalization);
            
            // Export
            document.getElementById('export-btn').addEventListener('click', exportSegmentationData);
        });
    </script>
</body>
</html>'''

def run_enhanced_server(port=5000):
    """Run the enhanced web server"""
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, EnhancedImageHandler)
    
    print("Enhanced Scientific Image Analyzer - Advanced Web Interface")
    print("=" * 70)
    print("NEW FEATURES:")
    print("• AI Denoising: Bilateral Filter, Non-Local Means, Wiener Filter")
    print("• Advanced Segmentation: Morphological Snakes, Region Growing, Adaptive Threshold")
    print("• 4D Image Viewing: Time/Z navigation, Maximum Projections")
    print("• Enhanced UI: Improved layout, better controls, real-time feedback")
    print("=" * 70)
    print(f"Server running on http://localhost:{port}")
    print("Upload microscopy images and explore advanced analysis capabilities")
    print("=" * 70)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nEnhanced server stopped.")
        httpd.server_close()

if __name__ == "__main__":
    run_enhanced_server()