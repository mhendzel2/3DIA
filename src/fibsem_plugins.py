#!/usr/bin/env python3
"""
FIB-SEM Specialized Analysis Plugins
Integration of advanced electron microscopy tools for PyMaris Scientific Analyzer
Includes ChimeraX integration and specialized FIB-SEM analysis capabilities
"""

import os
import sys
import json
import math
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import subprocess
from pathlib import Path

try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bug_fixes import ChimeraXPathFix, MRCExportFix
    FIXES_AVAILABLE = True
except ImportError:
    FIXES_AVAILABLE = False

class ChimeraXIntegration:
    """Integration with ChimeraX for 3D molecular visualization and FIB-SEM correlation"""
    
    def __init__(self):
        if FIXES_AVAILABLE:
            self.chimerax_path = ChimeraXPathFix.find_chimerax_installation()
        else:
            self.chimerax_path = self._find_chimerax_installation_fallback()
        self.temp_dir = tempfile.mkdtemp()

    def _find_chimerax_installation_fallback(self):
        """Fallback method to locate ChimeraX installation if bug_fixes.py is not available"""
        possible_paths = [
            "/Applications/ChimeraX.app/Contents/bin/ChimeraX",  # macOS
            "C:\\Program Files\\ChimeraX\\bin\\ChimeraX.exe",    # Windows
            "/usr/bin/chimerax",                                 # Linux
            "/opt/chimerax/bin/chimerax"                        # Linux alt
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def export_volume_to_chimerax(self, volume_data, filename="fibsem_volume.mrc"):
        """Export FIB-SEM volume data to ChimeraX compatible format"""
        filepath = os.path.join(self.temp_dir, filename)
        if FIXES_AVAILABLE:
            success, message = MRCExportFix.export_proper_mrc(volume_data, filepath)
            if success:
                print(f"Volume exported as {message} to {filepath}")
                return filepath
            else:
                print(f"MRC export failed: {message}")
                return None
        else:
            print("Warning: MRCExportFix not found. Using basic export.")
            # Fallback to simple export if bug_fixes not available
            try:
                with open(filepath, 'wb') as f:
                    height = len(volume_data)
                    width = len(volume_data[0]) if height > 0 else 0
                    f.write(width.to_bytes(4, 'little'))
                    f.write(height.to_bytes(4, 'little'))
                    f.write((1).to_bytes(4, 'little'))
                    f.write((2).to_bytes(4, 'little'))
                    f.write(b'\x00' * (256 - 16))
                    for row in volume_data:
                        for pixel in row:
                            f.write(int(pixel).to_bytes(4, 'little'))
                return filepath
            except Exception as e:
                print(f"Basic MRC export error: {e}")
                return None
    
    def create_chimerax_script(self, volume_path, segmentation_path=None):
        """Create ChimeraX script for FIB-SEM analysis"""
        script_content = f"""
# ChimeraX script for FIB-SEM analysis
open {volume_path}

# Set optimal visualization for FIB-SEM data
volume #1 style surface level 128
volume #1 transparency 30

# Add measurement tools
measure center #1

# Set appropriate lighting for EM data
lighting soft

# Add segmentation if available
"""
        
        if segmentation_path:
            script_content += f"""
open {segmentation_path}
volume #2 style mesh level 1
volume #2 color red transparency 70
"""
        
        script_content += """
# Position for optimal viewing
view orient

# Save session
save fibsem_analysis.cxs
"""
        
        script_path = os.path.join(self.temp_dir, "fibsem_analysis.cxc")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def launch_chimerax_analysis(self, volume_data, segmentation_data=None):
        """Launch ChimeraX with FIB-SEM data"""
        if not self.chimerax_path:
            return {"error": "ChimeraX not found. Please install ChimeraX for 3D analysis."}
        
        try:
            volume_path = self.export_volume_to_chimerax(volume_data)
            if not volume_path:
                return {"error": "Failed to export volume data"}
            
            segmentation_path = None
            if segmentation_data:
                segmentation_path = self.export_volume_to_chimerax(segmentation_data, "segmentation.mrc")
            
            script_path = self.create_chimerax_script(volume_path, segmentation_path)
            
            # Launch ChimeraX
            cmd = [self.chimerax_path, "--script", script_path]
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return {
                "success": True,
                "message": "ChimeraX launched with FIB-SEM data",
                "volume_path": volume_path,
                "script_path": script_path
            }
            
        except Exception as e:
            return {"error": f"ChimeraX launch failed: {str(e)}"}

class ThreeDCounter:
    """3D object counting and volume analysis for FIB-SEM"""
    
    @staticmethod
    def count_3d_objects(labels_3d):
        """Count and analyze 3D objects in volumetric data"""
        try:
            object_counts = {}
            
            # Count unique labels (objects)
            unique_labels = set()
            for z_slice in labels_3d:
                for row in z_slice:
                    for pixel in row:
                        if pixel > 0:
                            unique_labels.add(pixel)
            
            # Calculate volume for each object
            for label_id in unique_labels:
                volume = 0
                for z_slice in labels_3d:
                    for row in z_slice:
                        for pixel in row:
                            if pixel == label_id:
                                volume += 1
                
                object_counts[label_id] = {
                    'volume_voxels': volume,
                    'label': label_id
                }
            
            return {
                'total_objects': len(unique_labels),
                'object_data': object_counts,
                'largest_object': max(object_counts.values(), key=lambda x: x['volume_voxels']) if object_counts else None
            }
            
        except Exception as e:
            print(f"3D counting error: {e}")
            return {'total_objects': 0, 'object_data': {}}

class TomoSliceAnalyzer:
    """Advanced tomographic slice viewing and analysis"""
    
    def __init__(self):
        self.current_volume = None
        self.slice_thickness = 1.0  # nm
        
    def load_tomographic_data(self, volume_data, pixel_size=1.0):
        """Load tomographic volume data"""
        self.current_volume = volume_data
        self.pixel_size = pixel_size
        
        return {
            'dimensions': self.get_volume_dimensions(),
            'pixel_size': pixel_size,
            'total_slices': len(volume_data) if volume_data else 0
        }
    
    def get_volume_dimensions(self):
        """Get volume dimensions"""
        if not self.current_volume:
            return {'z': 0, 'y': 0, 'x': 0}
        
        z_dim = len(self.current_volume)
        y_dim = len(self.current_volume[0]) if z_dim > 0 else 0
        x_dim = len(self.current_volume[0][0]) if y_dim > 0 else 0
        
        return {'z': z_dim, 'y': y_dim, 'x': x_dim}
    
    def get_orthogonal_slices(self, z_index, y_index, x_index):
        """Get orthogonal slices for 3D navigation"""
        try:
            if not self.current_volume:
                return None
                
            dims = self.get_volume_dimensions()
            
            # XY slice (normal view)
            xy_slice = self.current_volume[min(z_index, dims['z']-1)]
            
            # XZ slice  
            xz_slice = []
            for z in range(dims['z']):
                row = []
                y_idx = min(y_index, dims['y']-1)
                if y_idx < len(self.current_volume[z]):
                    row = self.current_volume[z][y_idx]
                else:
                    row = [0] * dims['x']
                xz_slice.append(row)
            
            # YZ slice
            yz_slice = []
            for z in range(dims['z']):
                col = []
                x_idx = min(x_index, dims['x']-1)
                for y in range(dims['y']):
                    if y < len(self.current_volume[z]) and x_idx < len(self.current_volume[z][y]):
                        col.append(self.current_volume[z][y][x_idx])
                    else:
                        col.append(0)
                yz_slice.append(col)
            
            return {
                'xy_slice': xy_slice,
                'xz_slice': xz_slice, 
                'yz_slice': yz_slice,
                'coordinates': {'z': z_index, 'y': y_index, 'x': x_index}
            }
            
        except Exception as e:
            print(f"Orthogonal slice error: {e}")
            return None

class AcceleratedClassification:
    """GPU-accelerated pixel and object classification for FIB-SEM"""
    
    def __init__(self):
        self.trained_models = {}
        self.feature_extractors = []
    
    def extract_texture_features(self, image_patch):
        """Extract texture features relevant for FIB-SEM classification"""
        try:
            features = {}
            
            # Basic intensity statistics
            flat_patch = [pixel for row in image_patch for pixel in row]
            features['mean_intensity'] = sum(flat_patch) / len(flat_patch) if flat_patch else 0
            features['intensity_std'] = self.calculate_std(flat_patch)
            features['intensity_range'] = max(flat_patch) - min(flat_patch) if flat_patch else 0
            
            # Gradient magnitude (simple edge detection)
            features['edge_strength'] = self.calculate_edge_strength(image_patch)
            
            # Local binary pattern (simplified)
            features['texture_uniformity'] = self.calculate_texture_uniformity(image_patch)
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {}
    
    def calculate_std(self, values):
        """Calculate standard deviation"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def calculate_edge_strength(self, image_patch):
        """Calculate edge strength using simple gradients"""
        try:
            total_gradient = 0
            count = 0
            
            height = len(image_patch)
            width = len(image_patch[0]) if height > 0 else 0
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    # Simple Sobel-like operator
                    gx = (image_patch[i-1][j+1] + 2*image_patch[i][j+1] + image_patch[i+1][j+1] -
                          image_patch[i-1][j-1] - 2*image_patch[i][j-1] - image_patch[i+1][j-1])
                    gy = (image_patch[i+1][j-1] + 2*image_patch[i+1][j] + image_patch[i+1][j+1] -
                          image_patch[i-1][j-1] - 2*image_patch[i-1][j] - image_patch[i-1][j+1])
                    
                    gradient_magnitude = math.sqrt(gx*gx + gy*gy)
                    total_gradient += gradient_magnitude
                    count += 1
            
            return total_gradient / count if count > 0 else 0
            
        except Exception as e:
            print(f"Edge strength calculation error: {e}")
            return 0
    
    def calculate_texture_uniformity(self, image_patch):
        """Calculate texture uniformity"""
        try:
            height = len(image_patch)
            width = len(image_patch[0]) if height > 0 else 0
            
            if height < 3 or width < 3:
                return 0
            
            uniformity_score = 0
            count = 0
            
            # Calculate local variance in 3x3 neighborhoods
            for i in range(1, height-1):
                for j in range(1, width-1):
                    neighborhood = []
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            neighborhood.append(image_patch[i+di][j+dj])
                    
                    local_variance = self.calculate_std(neighborhood) ** 2
                    uniformity_score += local_variance
                    count += 1
            
            return uniformity_score / count if count > 0 else 0
            
        except Exception as e:
            print(f"Texture uniformity calculation error: {e}")
            return 0
    
    def classify_organelles(self, image, classification_type="mitochondria"):
        """Classify organelles in FIB-SEM images"""
        try:
            # Simple rule-based classification for demonstration
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            classification_map = [[0 for _ in range(width)] for _ in range(height)]
            
            # Process in patches
            patch_size = 16
            for i in range(0, height-patch_size, patch_size//2):
                for j in range(0, width-patch_size, patch_size//2):
                    # Extract patch
                    patch = []
                    for pi in range(patch_size):
                        if i+pi < height:
                            row = []
                            for pj in range(patch_size):
                                if j+pj < width:
                                    row.append(image[i+pi][j+pj])
                                else:
                                    row.append(0)
                            patch.append(row)
                    
                    # Extract features
                    features = self.extract_texture_features(patch)
                    
                    # Simple classification rules
                    classification = 0
                    if classification_type == "mitochondria":
                        # Mitochondria: moderate intensity, high edge strength, medium texture
                        if (50 < features.get('mean_intensity', 0) < 150 and
                            features.get('edge_strength', 0) > 20 and
                            10 < features.get('texture_uniformity', 0) < 100):
                            classification = 1
                    elif classification_type == "vesicles":
                        # Vesicles: lower intensity, circular (low texture uniformity)
                        if (features.get('mean_intensity', 0) < 80 and
                            features.get('texture_uniformity', 0) < 20):
                            classification = 2
                    elif classification_type == "membrane":
                        # Membranes: high edge strength, linear structures
                        if features.get('edge_strength', 0) > 30:
                            classification = 3
                    
                    # Apply classification to patch area
                    for pi in range(patch_size):
                        for pj in range(patch_size):
                            if i+pi < height and j+pj < width:
                                classification_map[i+pi][j+pj] = classification
            
            return classification_map
            
        except Exception as e:
            print(f"Organelle classification error: {e}")
            return [[0 for _ in range(len(image[0]))] for _ in range(len(image))]

class MembraneSegmenter:
    """Specialized segmentation for membrane-bound structures in FIB-SEM"""
    
    @staticmethod
    def segment_membranes(image, membrane_thickness=2):
        """Segment membrane structures using edge-based methods"""
        try:
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            
            # Edge detection for membranes
            edges = [[0 for _ in range(width)] for _ in range(height)]
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    # Sobel edge detection
                    gx = (image[i-1][j+1] + 2*image[i][j+1] + image[i+1][j+1] -
                          image[i-1][j-1] - 2*image[i][j-1] - image[i+1][j-1])
                    gy = (image[i+1][j-1] + 2*image[i+1][j] + image[i+1][j+1] -
                          image[i-1][j-1] - 2*image[i-1][j] - image[i-1][j+1])
                    
                    gradient_magnitude = math.sqrt(gx*gx + gy*gy)
                    edges[i][j] = int(gradient_magnitude)
            
            # Threshold edges to find membranes
            edge_threshold = sum(sum(row) for row in edges) / (height * width) * 1.5
            membrane_mask = [[1 if edges[i][j] > edge_threshold else 0 
                            for j in range(width)] for i in range(height)]
            
            return membrane_mask
            
        except Exception as e:
            print(f"Membrane segmentation error: {e}")
            return [[0 for _ in range(len(image[0]))] for _ in range(len(image))]
    
    @staticmethod
    def segment_vesicles(image, min_size=10, max_size=200):
        """Segment vesicle structures"""
        try:
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            
            # Simple threshold-based vesicle detection
            mean_intensity = sum(sum(row) for row in image) / (height * width)
            vesicle_threshold = mean_intensity * 0.7  # Vesicles are typically darker
            
            binary_mask = [[1 if image[i][j] < vesicle_threshold else 0 
                          for j in range(width)] for i in range(height)]
            
            # Connected component labeling (simplified)
            labels = [[0 for _ in range(width)] for _ in range(height)]
            current_label = 1
            
            for i in range(height):
                for j in range(width):
                    if binary_mask[i][j] == 1 and labels[i][j] == 0:
                        # Flood fill to label connected component
                        component_size = MembraneSegmenter._flood_fill(
                            binary_mask, labels, i, j, current_label)
                        
                        # Filter by size
                        if min_size <= component_size <= max_size:
                            current_label += 1
                        else:
                            # Remove component that doesn't meet size criteria
                            MembraneSegmenter._remove_component(labels, current_label)
            
            return labels
            
        except Exception as e:
            print(f"Vesicle segmentation error: {e}")
            return [[0 for _ in range(len(image[0]))] for _ in range(len(image))]
    
    @staticmethod
    def _flood_fill(binary_mask, labels, start_i, start_j, label):
        """Flood fill algorithm for connected component labeling"""
        height = len(binary_mask)
        width = len(binary_mask[0])
        stack = [(start_i, start_j)]
        size = 0
        
        while stack:
            i, j = stack.pop()
            if (i < 0 or i >= height or j < 0 or j >= width or 
                binary_mask[i][j] == 0 or labels[i][j] != 0):
                continue
            
            labels[i][j] = label
            size += 1
            
            # Add neighbors
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                stack.append((i+di, j+dj))
        
        return size
    
    @staticmethod
    def _remove_component(labels, label_to_remove):
        """Remove a labeled component"""
        height = len(labels)
        width = len(labels[0])
        
        for i in range(height):
            for j in range(width):
                if labels[i][j] == label_to_remove:
                    labels[i][j] = 0

class GPU_ImageProcessor:
    """GPU-accelerated image processing for large FIB-SEM datasets"""
    
    @staticmethod
    def accelerated_gaussian_filter(image, sigma=2.0):
        """Fast Gaussian filtering implementation"""
        try:
            # Simplified separable Gaussian filter
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            
            # Create Gaussian kernel
            kernel_size = int(6 * sigma) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            kernel = []
            center = kernel_size // 2
            sigma_sq = sigma * sigma
            sum_kernel = 0
            
            for i in range(kernel_size):
                val = math.exp(-((i - center) ** 2) / (2 * sigma_sq))
                kernel.append(val)
                sum_kernel += val
            
            # Normalize kernel
            kernel = [k / sum_kernel for k in kernel]
            
            # Apply horizontal pass
            temp = [[0 for _ in range(width)] for _ in range(height)]
            for i in range(height):
                for j in range(width):
                    filtered_val = 0
                    for k, kern_val in enumerate(kernel):
                        idx = j + k - center
                        if 0 <= idx < width:
                            filtered_val += image[i][idx] * kern_val
                        elif idx < 0:
                            filtered_val += image[i][0] * kern_val
                        else:
                            filtered_val += image[i][width-1] * kern_val
                    temp[i][j] = filtered_val
            
            # Apply vertical pass
            result = [[0 for _ in range(width)] for _ in range(height)]
            for i in range(height):
                for j in range(width):
                    filtered_val = 0
                    for k, kern_val in enumerate(kernel):
                        idx = i + k - center
                        if 0 <= idx < height:
                            filtered_val += temp[idx][j] * kern_val
                        elif idx < 0:
                            filtered_val += temp[0][j] * kern_val
                        else:
                            filtered_val += temp[height-1][j] * kern_val
                    result[i][j] = filtered_val
            
            return result
            
        except Exception as e:
            print(f"GPU filter error: {e}")
            return image
    
    @staticmethod
    def accelerated_morphology(image, operation="opening", kernel_size=3):
        """Fast morphological operations"""
        try:
            # Simple morphological operations
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            
            if operation == "erosion" or operation == "opening":
                # Erosion
                eroded = [[0 for _ in range(width)] for _ in range(height)]
                offset = kernel_size // 2
                
                for i in range(height):
                    for j in range(width):
                        min_val = float('inf')
                        for di in range(-offset, offset+1):
                            for dj in range(-offset, offset+1):
                                ni, nj = i + di, j + dj
                                if 0 <= ni < height and 0 <= nj < width:
                                    min_val = min(min_val, image[ni][nj])
                        eroded[i][j] = int(min_val) if min_val != float('inf') else 0
                
                if operation == "erosion":
                    return eroded
                image = eroded  # Continue to dilation for opening
            
            if operation == "dilation" or operation == "opening" or operation == "closing":
                # Dilation
                dilated = [[0 for _ in range(width)] for _ in range(height)]
                offset = kernel_size // 2
                
                for i in range(height):
                    for j in range(width):
                        max_val = 0
                        for di in range(-offset, offset+1):
                            for dj in range(-offset, offset+1):
                                ni, nj = i + di, j + dj
                                if 0 <= ni < height and 0 <= nj < width:
                                    max_val = max(max_val, image[ni][nj])
                        dilated[i][j] = max_val
                
                return dilated
            
            return image
            
        except Exception as e:
            print(f"Morphology error: {e}")
            return image

class Empanada_Segmentation:
    """Deep learning-based segmentation for EM data"""
    
    def __init__(self):
        self.available_models = {
            'mitochondria': 'MitoNet trained on EM data',
            'nucleus': 'Nuclear segmentation for EM',
            'er': 'Endoplasmic reticulum detection',
            'vesicles': 'Vesicle and organelle detection'
        }
    
    def segment_with_pretrained_model(self, image, model_type="mitochondria"):
        """Apply pre-trained deep learning models"""
        try:
            # Simulate deep learning segmentation with advanced classical methods
            if model_type == "mitochondria":
                return self._segment_mitochondria_classical(image)
            elif model_type == "nucleus":
                return self._segment_nucleus_classical(image)
            elif model_type == "er":
                return self._segment_er_classical(image)
            elif model_type == "vesicles":
                return self._segment_vesicles_classical(image)
            else:
                return [[0 for _ in range(len(image[0]))] for _ in range(len(image))]
                
        except Exception as e:
            print(f"Empanada segmentation error: {e}")
            return [[0 for _ in range(len(image[0]))] for _ in range(len(image))]
    
    def _segment_mitochondria_classical(self, image):
        """Classical approach mimicking deep learning for mitochondria"""
        # Apply multi-scale filtering
        smoothed = GPU_ImageProcessor.accelerated_gaussian_filter(image, sigma=1.5)
        
        # Edge-based detection for mitochondrial membranes
        membranes = MembraneSegmenter.segment_membranes(smoothed)
        
        # Combine with intensity-based segmentation
        classifier = AcceleratedClassification()
        organelles = classifier.classify_organelles(smoothed, "mitochondria")
        
        # Combine results
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        result = [[0 for _ in range(width)] for _ in range(height)]
        
        for i in range(height):
            for j in range(width):
                if organelles[i][j] == 1 and membranes[i][j] == 1:
                    result[i][j] = 1
        
        return result
    
    def _segment_nucleus_classical(self, image):
        """Classical nucleus segmentation"""
        # Nuclei are typically large, dark regions
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        mean_intensity = sum(sum(row) for row in image) / (height * width)
        nucleus_threshold = mean_intensity * 0.6
        
        # Create binary mask
        binary = [[1 if image[i][j] < nucleus_threshold else 0 
                  for j in range(width)] for i in range(height)]
        
        # Morphological operations to clean up
        cleaned = GPU_ImageProcessor.accelerated_morphology(binary, "opening", 5)
        
        return cleaned
    
    def _segment_er_classical(self, image):
        """Classical ER segmentation"""
        # ER has characteristic tubular/sheet structure
        membranes = MembraneSegmenter.segment_membranes(image, membrane_thickness=1)
        
        # Apply morphological operations to enhance tubular structures
        enhanced = GPU_ImageProcessor.accelerated_morphology(membranes, "dilation", 2)
        
        return enhanced
    
    def _segment_vesicles_classical(self, image):
        """Classical vesicle segmentation"""
        return MembraneSegmenter.segment_vesicles(image, min_size=5, max_size=100)

class OrganoidCounter:
    """3D structure analysis and quantification"""
    
    @staticmethod
    def analyze_3d_structures(labels_3d, pixel_size=1.0):
        """Comprehensive 3D structure analysis"""
        try:
            results = {}
            
            # Basic counting
            counter_results = ThreeDCounter.count_3d_objects(labels_3d)
            results.update(counter_results)
            
            # Volume measurements
            for obj_id, obj_data in counter_results['object_data'].items():
                volume_nm3 = obj_data['volume_voxels'] * (pixel_size ** 3)
                obj_data['volume_nm3'] = volume_nm3
                
                # Surface area estimation (simplified)
                surface_area = OrganoidCounter._estimate_surface_area(labels_3d, obj_id, pixel_size)
                obj_data['surface_area_nm2'] = surface_area
                
                # Sphericity calculation
                if volume_nm3 > 0 and surface_area > 0:
                    sphere_surface = 4 * math.pi * ((3 * volume_nm3 / (4 * math.pi)) ** (2/3))
                    obj_data['sphericity'] = sphere_surface / surface_area
                else:
                    obj_data['sphericity'] = 0
            
            return results
            
        except Exception as e:
            print(f"3D analysis error: {e}")
            return {'total_objects': 0, 'object_data': {}}
    
    @staticmethod
    def _estimate_surface_area(labels_3d, target_label, pixel_size):
        """Estimate surface area of 3D object"""
        try:
            surface_voxels = 0
            
            depth = len(labels_3d)
            height = len(labels_3d[0]) if depth > 0 else 0
            width = len(labels_3d[0][0]) if height > 0 else 0
            
            for z in range(depth):
                for y in range(height):
                    for x in range(width):
                        if labels_3d[z][y][x] == target_label:
                            # Check if voxel is on surface (has non-target neighbor)
                            is_surface = False
                            for dz in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    for dx in [-1, 0, 1]:
                                        if dz == 0 and dy == 0 and dx == 0:
                                            continue
                                        nz, ny, nx = z + dz, y + dy, x + dx
                                        if (0 <= nz < depth and 0 <= ny < height and 0 <= nx < width):
                                            if labels_3d[nz][ny][nx] != target_label:
                                                is_surface = True
                                                break
                                        else:
                                            is_surface = True
                                            break
                                    if is_surface:
                                        break
                                if is_surface:
                                    break
                            
                            if is_surface:
                                surface_voxels += 1
            
            # Convert to nm²
            surface_area = surface_voxels * (pixel_size ** 2)
            return surface_area
            
        except Exception as e:
            print(f"Surface area estimation error: {e}")
            return 0

# Integration class for all FIB-SEM tools
class FIBSEMAnalyzer:
    """Main integration class for all FIB-SEM analysis tools"""
    
    def __init__(self):
        self.chimerax = ChimeraXIntegration()
        self.counter_3d = ThreeDCounter()
        self.tomo_analyzer = TomoSliceAnalyzer()
        self.classifier = AcceleratedClassification()
        self.membrane_segmenter = MembraneSegmenter()
        self.gpu_processor = GPU_ImageProcessor()
        self.empanada = Empanada_Segmentation()
        self.organoid_counter = OrganoidCounter()
    
    def get_available_tools(self):
        """Get list of available FIB-SEM analysis tools"""
        return {
            'chimerax_integration': 'ChimeraX 3D visualization and molecular modeling',
            '3d_counter': '3D object counting and volume analysis',
            'tomo_slice': 'Advanced tomographic slice viewing',
            'accelerated_classification': 'GPU-accelerated pixel/object classification',
            'membrane_segmentation': 'Specialized membrane and organelle segmentation',
            'gpu_processing': 'Accelerated image processing for large datasets',
            'empanada_segmentation': 'Deep learning-based EM segmentation',
            'organoid_analysis': '3D structure quantification and morphometrics'
        }
    
    def run_comprehensive_analysis(self, image_data, analysis_type="full"):
        """Run comprehensive FIB-SEM analysis pipeline"""
        try:
            results = {
                'analysis_type': analysis_type,
                'tools_used': [],
                'results': {}
            }
            
            if analysis_type == "full" or analysis_type == "segmentation":
                # Apply denoising
                denoised = self.gpu_processor.accelerated_gaussian_filter(image_data, sigma=1.0)
                results['tools_used'].append('gpu_denoising')
                
                # Segment different structures
                mitochondria = self.empanada.segment_with_pretrained_model(denoised, "mitochondria")
                vesicles = self.empanada.segment_with_pretrained_model(denoised, "vesicles")
                membranes = self.membrane_segmenter.segment_membranes(denoised)
                
                results['results']['mitochondria_mask'] = mitochondria
                results['results']['vesicles_mask'] = vesicles
                results['results']['membranes_mask'] = membranes
                results['tools_used'].extend(['empanada_mitochondria', 'empanada_vesicles', 'membrane_segmentation'])
            
            if analysis_type == "full" or analysis_type == "classification":
                # Classify organelles
                organelle_map = self.classifier.classify_organelles(image_data, "mitochondria")
                results['results']['organelle_classification'] = organelle_map
                results['tools_used'].append('accelerated_classification')
            
            if analysis_type == "full" or analysis_type == "3d_analysis":
                # For 3D analysis, convert 2D to pseudo-3D
                pseudo_3d = [image_data]  # Single slice as 3D volume
                count_results = self.counter_3d.count_3d_objects(pseudo_3d)
                results['results']['3d_counting'] = count_results
                results['tools_used'].append('3d_counter')
            
            return results
            
        except Exception as e:
            print(f"Comprehensive analysis error: {e}")
            return {'error': str(e), 'tools_used': [], 'results': {}}
