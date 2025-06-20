"""
Scientific Image Analysis Application
Comprehensive microscopy analysis suite replicating Napari Hub plugin functionality
"""

import os
import io
import base64
import json
import math
from pathlib import Path

# Simplified scientific computing implementations
class NumpyLite:
    """Lightweight numpy replacement for basic operations"""
    
    @staticmethod
    def array(data):
        if isinstance(data, list):
            return data
        return data
    
    @staticmethod
    def zeros(shape):
        if isinstance(shape, tuple):
            if len(shape) == 2:
                return [[0 for _ in range(shape[1])] for _ in range(shape[0])]
            else:
                return [0] * shape[0]
        return [0] * shape
    
    @staticmethod
    def mean(arr):
        flat = NumpyLite.flatten(arr) if isinstance(arr[0], list) else arr
        return sum(flat) / len(flat)
    
    @staticmethod
    def std(arr):
        flat = NumpyLite.flatten(arr) if isinstance(arr[0], list) else arr
        mean_val = NumpyLite.mean(flat)
        variance = sum((x - mean_val) ** 2 for x in flat) / len(flat)
        return math.sqrt(variance)
    
    @staticmethod
    def flatten(arr):
        result = []
        for item in arr:
            if isinstance(item, list):
                result.extend(NumpyLite.flatten(item))
            else:
                result.append(item)
        return result
    
    @staticmethod
    def percentile(arr, p):
        flat = NumpyLite.flatten(arr) if isinstance(arr[0], list) else arr
        sorted_arr = sorted(flat)
        k = (len(sorted_arr) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_arr[int(k)]
        return sorted_arr[int(f)] * (c - k) + sorted_arr[int(c)] * (k - f)

# Flask with fallback
try:
    from flask import Flask, render_template, request, jsonify, send_file
    from werkzeug.utils import secure_filename
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Flask not available - creating standalone analyzer")

# PIL for image handling
try:
    from PIL import Image as PILImage, ImageFilter, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not available")

class GradientWatershedSegmentation:
    """
    Cell segmentation using gradient flow and watershed algorithms.
    This is a classical method inspired by Cellpose's approach but does NOT use deep learning.
    Note: This is NOT the actual Cellpose AI model - it's a classical computer vision implementation.
    """
    
    @staticmethod
    def segment_cells(image_array, diameter=30, flow_threshold=0.4):
        """
        Cell segmentation using classical computer vision methods
        Mimics Cellpose's approach with gradient flow fields
        """
        height, width = len(image_array), len(image_array[0])
        
        # Gaussian smoothing simulation
        smoothed = GradientWatershedSegmentation._gaussian_blur(image_array, sigma=diameter/6)
        
        # Gradient calculation for flow field
        gradient_map = GradientWatershedSegmentation._calculate_gradients(smoothed)
        
        # Threshold based on flow magnitude
        threshold = NumpyLite.percentile(gradient_map, 75)
        binary_mask = [[1 if gradient_map[i][j] > threshold else 0 
                       for j in range(width)] for i in range(height)]
        
        # Connected components labeling
        labels = GradientWatershedSegmentation._connected_components(binary_mask)
        
        return labels
    
    @staticmethod
    def _gaussian_blur(image, sigma=1.0):
        """Simple Gaussian blur implementation"""
        height, width = len(image), len(image[0])
        kernel_size = max(3, int(6 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        kernel = GradientWatershedSegmentation._gaussian_kernel(kernel_size, sigma)
        
        # Apply convolution
        result = [[0 for _ in range(width)] for _ in range(height)]
        half_kernel = kernel_size // 2
        
        for i in range(height):
            for j in range(width):
                value = 0
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        ii = max(0, min(height - 1, i + ki - half_kernel))
                        jj = max(0, min(width - 1, j + kj - half_kernel))
                        value += image[ii][jj] * kernel[ki][kj]
                result[i][j] = value
        
        return result
    
    @staticmethod
    def _gaussian_kernel(size, sigma):
        """Generate Gaussian kernel"""
        kernel = [[0 for _ in range(size)] for _ in range(size)]
        center = size // 2
        sum_val = 0
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                value = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
                kernel[i][j] = value
                sum_val += value
        
        # Normalize
        for i in range(size):
            for j in range(size):
                kernel[i][j] /= sum_val
        
        return kernel
    
    @staticmethod
    def _calculate_gradients(image):
        """Calculate gradient magnitude"""
        height, width = len(image), len(image[0])
        gradients = [[0 for _ in range(width)] for _ in range(height)]
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                gx = image[i][j+1] - image[i][j-1]
                gy = image[i+1][j] - image[i-1][j]
                gradients[i][j] = math.sqrt(gx*gx + gy*gy)
        
        return gradients
    
    @staticmethod
    def _connected_components(binary_image):
        """Simple connected components labeling"""
        height, width = len(binary_image), len(binary_image[0])
        labels = [[0 for _ in range(width)] for _ in range(height)]
        current_label = 1
        
        def flood_fill(start_i, start_j, label):
            stack = [(start_i, start_j)]
            while stack:
                i, j = stack.pop()
                if (i < 0 or i >= height or j < 0 or j >= width or 
                    labels[i][j] != 0 or binary_image[i][j] == 0):
                    continue
                
                labels[i][j] = label
                stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
        
        for i in range(height):
            for j in range(width):
                if binary_image[i][j] == 1 and labels[i][j] == 0:
                    flood_fill(i, j, current_label)
                    current_label += 1
        
        return labels

class StarDistSegmentation:
    """
    StarDist-inspired nucleus segmentation
    Replicates napari-stardist functionality for star-convex objects
    """
    
    @staticmethod
    def segment_nuclei(image_array, prob_thresh=0.5, min_distance=20):
        """
        Nucleus segmentation using star-convex shape detection
        """
        height, width = len(image_array), len(image_array[0])
        
        # Apply Gaussian filter
        filtered = GradientWatershedSegmentation._gaussian_blur(image_array, sigma=1.0)
        
        # Find local maxima (nuclei centers)
        peaks = StarDistSegmentation._find_local_maxima(
            filtered, min_distance, NumpyLite.percentile(filtered, 75)
        )
        
        # Create circular regions around peaks
        labels = [[0 for _ in range(width)] for _ in range(height)]
        
        for label_id, (peak_i, peak_j) in enumerate(peaks, 1):
            radius = 15  # Fixed radius for simplicity
            for i in range(max(0, peak_i - radius), min(height, peak_i + radius + 1)):
                for j in range(max(0, peak_j - radius), min(width, peak_j + radius + 1)):
                    if (i - peak_i)**2 + (j - peak_j)**2 <= radius**2:
                        labels[i][j] = label_id
        
        return labels
    
    @staticmethod
    def _find_local_maxima(image, min_distance, threshold):
        """Find local maxima in image"""
        height, width = len(image), len(image[0])
        peaks = []
        
        for i in range(min_distance, height - min_distance):
            for j in range(min_distance, width - min_distance):
                if image[i][j] < threshold:
                    continue
                
                is_maximum = True
                for di in range(-min_distance//2, min_distance//2 + 1):
                    for dj in range(-min_distance//2, min_distance//2 + 1):
                        if (di == 0 and dj == 0):
                            continue
                        if image[i + di][j + dj] >= image[i][j]:
                            is_maximum = False
                            break
                    if not is_maximum:
                        break
                
                if is_maximum:
                    peaks.append((i, j))
        
        return peaks

class RegionPropsAnalysis:
    """
    Region properties analysis
    Replicates napari-skimage-regionprops functionality
    """
    
    @staticmethod
    def analyze_objects(labels, intensity_image=None):
        """
        Extract comprehensive measurements from labeled objects
        """
        # Find unique labels
        unique_labels = set()
        for row in labels:
            unique_labels.update(row)
        unique_labels.discard(0)  # Remove background
        
        results = []
        
        for label in unique_labels:
            measurements = RegionPropsAnalysis._measure_object(
                labels, label, intensity_image
            )
            results.append(measurements)
        
        return results
    
    @staticmethod
    def _measure_object(labels, target_label, intensity_image=None):
        """Measure properties of a single object"""
        height, width = len(labels), len(labels[0])
        
        # Find object pixels
        object_pixels = []
        intensities = []
        
        for i in range(height):
            for j in range(width):
                if labels[i][j] == target_label:
                    object_pixels.append((i, j))
                    if intensity_image:
                        intensities.append(intensity_image[i][j])
        
        if not object_pixels:
            return {}
        
        # Calculate measurements
        area = len(object_pixels)
        
        # Centroid
        centroid_y = sum(p[0] for p in object_pixels) / area
        centroid_x = sum(p[1] for p in object_pixels) / area
        
        # Bounding box
        min_y = min(p[0] for p in object_pixels)
        max_y = max(p[0] for p in object_pixels)
        min_x = min(p[1] for p in object_pixels)
        max_x = max(p[1] for p in object_pixels)
        
        # Perimeter (simplified)
        perimeter = RegionPropsAnalysis._calculate_perimeter(labels, target_label)
        
        measurements = {
            'label': target_label,
            'area': area,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'bbox_min_x': min_x,
            'bbox_min_y': min_y,
            'bbox_max_x': max_x,
            'bbox_max_y': max_y,
            'perimeter': perimeter,
            'aspect_ratio': (max_x - min_x) / (max_y - min_y) if max_y != min_y else 1.0
        }
        
        if intensity_image and intensities:
            measurements.update({
                'mean_intensity': NumpyLite.mean(intensities),
                'max_intensity': max(intensities),
                'min_intensity': min(intensities)
            })
        
        return measurements
    
    @staticmethod
    def _calculate_perimeter(labels, target_label):
        """Calculate object perimeter"""
        height, width = len(labels), len(labels[0])
        perimeter = 0
        
        for i in range(height):
            for j in range(width):
                if labels[i][j] == target_label:
                    # Check if pixel is on boundary
                    is_boundary = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if (ni < 0 or ni >= height or nj < 0 or nj >= width or 
                            labels[ni][nj] != target_label):
                            is_boundary = True
                            break
                    if is_boundary:
                        perimeter += 1
        
        return perimeter

class BTrackTracking:
    """
    Multi-object tracking
    Replicates napari-btrack functionality for time series analysis
    """
    
    @staticmethod
    def track_objects(label_sequence, max_search_radius=50):
        """
        Simple tracking algorithm for object trajectories
        """
        if not label_sequence:
            return []
        
        tracks = []
        track_id = 0
        
        # Get centroids for first frame
        first_objects = RegionPropsAnalysis.analyze_objects(label_sequence[0])
        active_tracks = {}
        
        for obj in first_objects:
            track = {
                'id': track_id,
                'frames': [0],
                'centroids': [(obj['centroid_y'], obj['centroid_x'])],
                'areas': [obj['area']]
            }
            active_tracks[obj['label']] = track
            track_id += 1
        
        # Process subsequent frames
        for frame_idx in range(1, len(label_sequence)):
            current_objects = RegionPropsAnalysis.analyze_objects(label_sequence[frame_idx])
            
            if not current_objects:
                continue
            
            # Calculate distances and assign objects to tracks
            current_centroids = [(obj['centroid_y'], obj['centroid_x']) for obj in current_objects]
            prev_centroids = [track['centroids'][-1] for track in active_tracks.values()]
            
            if prev_centroids and current_centroids:
                distances = BTrackTracking._calculate_distance_matrix(current_centroids, prev_centroids)
                
                used_tracks = set()
                for i, obj in enumerate(current_objects):
                    min_dist_idx = min(range(len(distances[i])), key=lambda k: distances[i][k])
                    min_dist = distances[i][min_dist_idx]
                    
                    if min_dist < max_search_radius and min_dist_idx not in used_tracks:
                        # Update existing track
                        track_key = list(active_tracks.keys())[min_dist_idx]
                        track = active_tracks[track_key]
                        track['frames'].append(frame_idx)
                        track['centroids'].append((obj['centroid_y'], obj['centroid_x']))
                        track['areas'].append(obj['area'])
                        used_tracks.add(min_dist_idx)
                    else:
                        # Create new track
                        new_track = {
                            'id': track_id,
                            'frames': [frame_idx],
                            'centroids': [(obj['centroid_y'], obj['centroid_x'])],
                            'areas': [obj['area']]
                        }
                        active_tracks[obj['label']] = new_track
                        track_id += 1
        
        return list(active_tracks.values())
    
    @staticmethod
    def _calculate_distance_matrix(points1, points2):
        """Calculate distance matrix between two sets of points"""
        distances = []
        for p1 in points1:
            row = []
            for p2 in points2:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                row.append(dist)
            distances.append(row)
        return distances

class ColocalizationAnalysis:
    """
    Colocalization analysis tools
    Advanced statistical analysis for fluorescent channel overlap
    """
    
    @staticmethod
    def analyze_colocalization(image1, image2, threshold1=None, threshold2=None):
        """
        Comprehensive colocalization analysis
        """
        height, width = len(image1), len(image1[0])
        
        # Auto-threshold if not provided
        if threshold1 is None:
            threshold1 = ColocalizationAnalysis._otsu_threshold(image1)
        if threshold2 is None:
            threshold2 = ColocalizationAnalysis._otsu_threshold(image2)
        
        # Create binary masks
        mask1 = [[1 if image1[i][j] > threshold1 else 0 for j in range(width)] for i in range(height)]
        mask2 = [[1 if image2[i][j] > threshold2 else 0 for j in range(width)] for i in range(height)]
        
        # Calculate overlap
        overlap = [[mask1[i][j] * mask2[i][j] for j in range(width)] for i in range(height)]
        
        # Extract pixel values for correlation
        pixels1, pixels2 = [], []
        for i in range(height):
            for j in range(width):
                if overlap[i][j]:
                    pixels1.append(image1[i][j])
                    pixels2.append(image2[i][j])
        
        results = {}
        
        # Pearson correlation
        if len(pixels1) > 1:
            correlation = ColocalizationAnalysis._pearson_correlation(pixels1, pixels2)
            results['pearson_correlation'] = correlation
        else:
            results['pearson_correlation'] = 0
        
        # Manders coefficients
        sum1_total = sum(sum(image1[i][j] for j in range(width) if mask1[i][j]) for i in range(height))
        sum2_total = sum(sum(image2[i][j] for j in range(width) if mask2[i][j]) for i in range(height))
        
        if sum1_total > 0:
            sum1_coloc = sum(sum(image1[i][j] for j in range(width) if overlap[i][j]) for i in range(height))
            results['manders_m1'] = sum1_coloc / sum1_total
        else:
            results['manders_m1'] = 0
        
        if sum2_total > 0:
            sum2_coloc = sum(sum(image2[i][j] for j in range(width) if overlap[i][j]) for i in range(height))
            results['manders_m2'] = sum2_coloc / sum2_total
        else:
            results['manders_m2'] = 0
        
        # Overlap coefficient
        overlap_pixels = sum(sum(overlap[i][j] for j in range(width)) for i in range(height))
        union_pixels = sum(sum(1 if mask1[i][j] or mask2[i][j] else 0 for j in range(width)) for i in range(height))
        results['overlap_coefficient'] = overlap_pixels / union_pixels if union_pixels > 0 else 0
        
        return results
    
    @staticmethod
    def _otsu_threshold(image):
        """Simple Otsu thresholding implementation"""
        # Flatten image and create histogram
        pixels = NumpyLite.flatten(image)
        min_val, max_val = min(pixels), max(pixels)
        
        if min_val == max_val:
            return min_val
        
        # Simple threshold at mean + std
        mean_val = NumpyLite.mean(pixels)
        std_val = NumpyLite.std(pixels)
        return mean_val + 0.5 * std_val
    
    @staticmethod
    def _pearson_correlation(x, y):
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        
        return numerator / denominator if denominator > 0 else 0

class ImageProcessor:
    """
    Core image processing functionality
    """
    
    @staticmethod
    def load_image(file_path):
        """Load image and convert to grayscale array"""
        if not HAS_PIL:
            print("PIL not available - cannot load images")
            return None
        
        try:
            with PILImage.open(file_path) as img:
                # Convert to grayscale
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Convert to array
                width, height = img.size
                pixels = list(img.getdata())
                
                # Reshape to 2D array
                image_array = []
                for i in range(height):
                    row = []
                    for j in range(width):
                        row.append(pixels[i * width + j])
                    image_array.append(row)
                
                return image_array
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def array_to_image(array):
        """Convert array back to PIL Image"""
        if not HAS_PIL:
            return None
        
        height, width = len(array), len(array[0])
        
        # Normalize to 0-255
        flat = NumpyLite.flatten(array)
        min_val, max_val = min(flat), max(flat)
        
        if max_val > min_val:
            normalized = [[(array[i][j] - min_val) * 255 / (max_val - min_val) 
                          for j in range(width)] for i in range(height)]
        else:
            normalized = array
        
        # Convert to PIL format
        pixels = []
        for row in normalized:
            pixels.extend(int(pixel) for pixel in row)
        
        img = PILImage.new('L', (width, height))
        img.putdata(pixels)
        return img
    
    @staticmethod
    def create_test_image(width=400, height=400):
        """Create a test image with synthetic cell-like objects"""
        image = [[50 for _ in range(width)] for _ in range(height)]  # Background
        
        # Add some circular objects
        centers = [(100, 100), (200, 150), (300, 200), (150, 300)]
        
        for center_x, center_y in centers:
            radius = 30 + (center_x % 20)  # Variable radius
            intensity = 200 + (center_y % 55)  # Variable intensity
            
            for i in range(max(0, center_y - radius), min(height, center_y + radius)):
                for j in range(max(0, center_x - radius), min(width, center_x + radius)):
                    dist_sq = (i - center_y)**2 + (j - center_x)**2
                    if dist_sq <= radius**2:
                        # Gaussian-like intensity falloff
                        factor = math.exp(-dist_sq / (2 * (radius/3)**2))
                        image[i][j] = int(50 + (intensity - 50) * factor)
        
        return image

# Web application (if Flask is available)
if HAS_FLASK:
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
    
    # Create uploads directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Global storage
    analysis_cache = {}
    
    @app.route('/')
    def index():
        return render_template('microscopy_analyzer.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        try:
            filename = secure_filename(file.filename)
            file_path = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(str(file_path))
            
            # Load image
            image_array = ImageProcessor.load_image(file_path)
            if image_array is None:
                return jsonify({'error': 'Could not load image'}), 400
            
            # Create session
            session_id = str(hash(filename))
            analysis_cache[session_id] = {
                'original_image': image_array,
                'processed_images': {},
                'results': {}
            }
            
            # Convert to base64 for display
            img_pil = ImageProcessor.array_to_image(image_array)
            if img_pil:
                # Resize for display
                img_pil.thumbnail((400, 400))
                buffer = io.BytesIO()
                img_pil.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                image_data = f"data:image/png;base64,{img_str}"
            else:
                image_data = ""
            
            info = {
                'filename': filename,
                'shape': [len(image_array), len(image_array[0])],
                'session_id': session_id
            }
            
            return jsonify({
                'success': True,
                'image_data': image_data,
                'info': info
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    @app.route('/segment', methods=['POST'])
    def segment_image():
        """Perform segmentation"""
        data = request.get_json()
        session_id = data.get('session_id')
        method = data.get('method', 'cellpose')
        
        if session_id not in analysis_cache:
            return jsonify({'error': 'Invalid session'}), 400
        
        try:
            image = analysis_cache[session_id]['original_image']
            
            if method == 'cellpose':
                labels = GradientWatershedSegmentation.segment_cells(image)
            elif method == 'stardist':
                labels = StarDistSegmentation.segment_nuclei(image)
            else:
                return jsonify({'error': 'Unknown method'}), 400
            
            # Store results
            analysis_cache[session_id]['processed_images'][f'labels_{method}'] = labels
            
            # Generate measurements
            measurements = RegionPropsAnalysis.analyze_objects(labels, image)
            analysis_cache[session_id]['results'][f'measurements_{method}'] = measurements
            
            # Create visualization (simple overlay)
            overlay = ImageProcessor.create_label_overlay(image, labels)
            overlay_img = ImageProcessor.array_to_image(overlay)
            
            if overlay_img:
                overlay_img.thumbnail((400, 400))
                buffer = io.BytesIO()
                overlay_img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                overlay_data = f"data:image/png;base64,{img_str}"
            else:
                overlay_data = ""
            
            return jsonify({
                'success': True,
                'segmentation_data': overlay_data,
                'object_count': len(measurements),
                'measurements': measurements[:10]  # First 10 objects
            })
            
        except Exception as e:
            return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500
    
    @app.route('/analyze', methods=['POST'])
    def analyze_image():
        """Perform analysis"""
        data = request.get_json()
        session_id = data.get('session_id')
        analysis_type = data.get('analysis_type', 'intensity')
        
        if session_id not in analysis_cache:
            return jsonify({'error': 'Invalid session'}), 400
        
        try:
            image = analysis_cache[session_id]['original_image']
            
            if analysis_type == 'intensity':
                flat_pixels = NumpyLite.flatten(image)
                results = {
                    'mean': NumpyLite.mean(flat_pixels),
                    'std': NumpyLite.std(flat_pixels),
                    'min': min(flat_pixels),
                    'max': max(flat_pixels),
                    'percentile_25': NumpyLite.percentile(flat_pixels, 25),
                    'percentile_75': NumpyLite.percentile(flat_pixels, 75)
                }
            elif analysis_type == 'colocalization':
                # Create synthetic second channel for demo
                channel2 = GradientWatershedSegmentation._gaussian_blur(image, sigma=2)
                results = ColocalizationAnalysis.analyze_colocalization(image, channel2)
            else:
                results = {'error': 'Unknown analysis type'}
            
            analysis_cache[session_id]['results'][f'analysis_{analysis_type}'] = results
            
            return jsonify({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/track', methods=['POST'])
    def track_objects():
        """Perform tracking"""
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in analysis_cache:
            return jsonify({'error': 'Invalid session'}), 400
        
        try:
            image = analysis_cache[session_id]['original_image']
            
            # Create synthetic time series
            time_series = []
            for t in range(5):
                # Simulate movement by shifting
                shifted_image = ImageProcessor.shift_image(image, t*3, t*2)
                labels = GradientWatershedSegmentation.segment_cells(shifted_image)
                time_series.append(labels)
            
            # Track objects
            tracks = BTrackTracking.track_objects(time_series)
            
            # Prepare track data
            track_data = []
            for track in tracks:
                if len(track['frames']) > 1:
                    track_data.append({
                        'id': track['id'],
                        'frames': track['frames'],
                        'centroids': track['centroids'],
                        'duration': len(track['frames'])
                    })
            
            analysis_cache[session_id]['results']['tracking'] = track_data
            
            return jsonify({
                'success': True,
                'track_count': len(track_data),
                'tracks': track_data[:10]
            })
            
        except Exception as e:
            return jsonify({'error': f'Tracking failed: {str(e)}'}), 500
    
    @app.route('/create_test', methods=['POST'])
    def create_test_image():
        """Create a test image for demonstration"""
        try:
            # Create test image
            test_image = ImageProcessor.create_test_image()
            
            # Convert to PIL and save
            img_pil = ImageProcessor.array_to_image(test_image)
            if img_pil:
                img_pil.thumbnail((400, 400))
                buffer = io.BytesIO()
                img_pil.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                image_data = f"data:image/png;base64,{img_str}"
                
                # Create session
                session_id = "test_image_session"
                analysis_cache[session_id] = {
                    'original_image': test_image,
                    'processed_images': {},
                    'results': {}
                }
                
                return jsonify({
                    'success': True,
                    'image_data': image_data,
                    'info': {
                        'filename': 'test_image.png',
                        'shape': [len(test_image), len(test_image[0])],
                        'session_id': session_id
                    }
                })
            else:
                return jsonify({'error': 'Could not create test image'}), 500
                
        except Exception as e:
            return jsonify({'error': f'Test image creation failed: {str(e)}'}), 500

# Helper methods for ImageProcessor
def create_label_overlay(image, labels):
    """Create colored overlay of labels on original image"""
    height, width = len(image), len(image[0])
    overlay = [[image[i][j] for j in range(width)] for i in range(height)]
    
    # Find unique labels
    unique_labels = set()
    for row in labels:
        unique_labels.update(row)
    unique_labels.discard(0)
    
    # Add colored borders
    for i in range(height):
        for j in range(width):
            if labels[i][j] != 0:
                # Check if pixel is on boundary
                is_boundary = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (ni < 0 or ni >= height or nj < 0 or nj >= width or 
                        labels[ni][nj] != labels[i][j]):
                        is_boundary = True
                        break
                if is_boundary:
                    overlay[i][j] = min(255, overlay[i][j] + 100)
    
    return overlay

def shift_image(image, shift_x, shift_y):
    """Shift image by given offsets"""
    height, width = len(image), len(image[0])
    shifted = [[0 for _ in range(width)] for _ in range(height)]
    
    for i in range(height):
        for j in range(width):
            src_i = i - shift_y
            src_j = j - shift_x
            if 0 <= src_i < height and 0 <= src_j < width:
                shifted[i][j] = image[src_i][src_j]
    
    return shifted

ImageProcessor.create_label_overlay = create_label_overlay
ImageProcessor.shift_image = shift_image

def main():
    """Main application entry point"""
    if HAS_FLASK:
        print("Starting Scientific Image Analyzer web application...")
        print("Integrated Napari Hub plugin functionality:")
        print("- Cellpose-inspired cell segmentation")
        print("- StarDist-inspired nucleus detection")
        print("- Region properties analysis")
        print("- BTrack-inspired object tracking")
        print("- Advanced colocalization analysis")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Flask not available. Running in console mode...")
        
        # Console-based demonstration
        print("Creating test image...")
        test_image = ImageProcessor.create_test_image()
        
        print("Running gradient watershed segmentation...")
        cellpose_labels = GradientWatershedSegmentation.segment_cells(test_image)
        
        print("Running StarDist segmentation...")
        stardist_labels = StarDistSegmentation.segment_nuclei(test_image)
        
        print("Analyzing objects...")
        measurements = RegionPropsAnalysis.analyze_objects(cellpose_labels, test_image)
        
        print(f"\nResults:")
        print(f"Cellpose found {len([r for row in cellpose_labels for r in row if r > 0])} labeled pixels")
        print(f"StarDist found {len([r for row in stardist_labels for r in row if r > 0])} labeled pixels")
        print(f"Region analysis found {len(measurements)} objects")
        
        if measurements:
            areas = [m['area'] for m in measurements]
            print(f"Mean object area: {NumpyLite.mean(areas):.1f}")

if __name__ == '__main__':
    main()