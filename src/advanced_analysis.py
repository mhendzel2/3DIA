#!/usr/bin/env python3
"""
Advanced Analysis Module for Scientific Image Analyzer
Includes 4D viewing, advanced segmentation, and AI denoising
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import math

# Fallback implementations when scientific libraries aren't available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Basic numpy-like operations
    class np:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def zeros_like(arr, dtype=int):
            if isinstance(arr, list):
                return [[0 for _ in row] for row in arr]
            return arr
        @staticmethod
        def ogrid(*args):
            return args
        @staticmethod
        def sqrt(x):
            return math.sqrt(x) if isinstance(x, (int, float)) else x
        @staticmethod
        def exp(x):
            return math.exp(x) if isinstance(x, (int, float)) else x
        @staticmethod
        def sum(arr):
            if isinstance(arr, list):
                return sum(sum(row) if isinstance(row, list) else row for row in arr)
            return arr
        @staticmethod
        def mean(arr):
            if isinstance(arr, list):
                flat = [item for sublist in arr for item in (sublist if isinstance(sublist, list) else [sublist])]
                return sum(flat) / len(flat) if flat else 0
            return arr
        @staticmethod
        def std(arr):
            if isinstance(arr, list):
                flat = [item for sublist in arr for item in (sublist if isinstance(sublist, list) else [sublist])]
                if not flat:
                    return 0
                mean_val = sum(flat) / len(flat)
                variance = sum((x - mean_val) ** 2 for x in flat) / len(flat)
                return math.sqrt(variance)
            return 0
        @staticmethod
        def max(arr):
            if isinstance(arr, list):
                flat = [item for sublist in arr for item in (sublist if isinstance(sublist, list) else [sublist])]
                return max(flat) if flat else 0
            return arr
        @staticmethod
        def min(arr):
            if isinstance(arr, list):
                flat = [item for sublist in arr for item in (sublist if isinstance(sublist, list) else [sublist])]
                return min(flat) if flat else 0
            return arr

class AdvancedSegmentation:
    """Advanced segmentation algorithms beyond basic methods"""
    
    @staticmethod
    def morphological_snakes(image, iterations: int = 100, smoothing: float = 1.0):
        """
        Morphological Active Contours (Snakes) segmentation
        Based on morphological operations for boundary detection
        """
        try:
            # Normalize image
            normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            # Initial contour - circular mask in center
            height, width = normalized.shape
            center_y, center_x = height // 2, width // 2
            radius = min(height, width) // 4
            
            # Create initial circular mask
            y, x = np.ogrid[:height, :width]
            mask = (y - center_y)**2 + (x - center_x)**2 <= radius**2
            
            # Iterative morphological evolution
            for i in range(iterations):
                # Gradient magnitude
                grad_y = np.gradient(normalized, axis=0)
                grad_x = np.gradient(normalized, axis=1)
                gradient_mag = np.sqrt(grad_y**2 + grad_x**2)
                
                # Morphological operations
                from scipy import ndimage
                dilated = ndimage.binary_dilation(mask)
                eroded = ndimage.binary_erosion(mask)
                
                # Update based on gradient information
                boundary = dilated ^ eroded
                edge_strength = np.mean(gradient_mag[boundary])
                
                if edge_strength > 0.1:  # Strong edge - contract
                    mask = eroded
                else:  # Weak edge - expand
                    mask = dilated
                
                # Apply smoothing
                if smoothing > 0:
                    mask = ndimage.gaussian_filter(mask.astype(float), sigma=smoothing) > 0.5
            
            return mask.astype(int)
            
        except Exception as e:
            print(f"Morphological snakes error: {e}")
            # Fallback to simple threshold
            threshold = np.mean(image) + np.std(image)
            return (image > threshold).astype(int)
    
    @staticmethod
    def region_growing(image: np.ndarray, seed_points: List[Tuple[int, int]], 
                      threshold: float = 0.1) -> np.ndarray:
        """
        Region growing segmentation from seed points
        """
        try:
            height, width = image.shape
            segmented = np.zeros((height, width), dtype=int)
            
            for label, (seed_y, seed_x) in enumerate(seed_points, 1):
                if seed_y >= height or seed_x >= width:
                    continue
                    
                visited = np.zeros((height, width), dtype=bool)
                stack = [(seed_y, seed_x)]
                seed_value = float(image[seed_y, seed_x])
                
                while stack:
                    y, x = stack.pop()
                    
                    if (y < 0 or y >= height or x < 0 or x >= width or 
                        visited[y, x] or segmented[y, x] != 0):
                        continue
                    
                    if abs(float(image[y, x]) - seed_value) <= threshold * 255:
                        visited[y, x] = True
                        segmented[y, x] = label
                        
                        # Add neighbors
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((y + dy, x + dx))
            
            return segmented
            
        except Exception as e:
            print(f"Region growing error: {e}")
            return np.zeros_like(image, dtype=int)
    
    @staticmethod
    def adaptive_threshold_segmentation(image: np.ndarray, block_size: int = 11, 
                                      offset: float = 2.0) -> np.ndarray:
        """
        Adaptive thresholding for varying illumination
        """
        try:
            # Local mean calculation
            from scipy import ndimage
            kernel = np.ones((block_size, block_size)) / (block_size * block_size)
            local_mean = ndimage.convolve(image.astype(float), kernel, mode='reflect')
            
            # Adaptive threshold
            threshold = local_mean - offset
            segmented = image > threshold
            
            # Label connected components
            labeled, num_features = ndimage.label(segmented)
            return labeled
            
        except Exception as e:
            print(f"Adaptive threshold error: {e}")
            # Fallback to global threshold
            threshold = np.mean(image)
            return (image > threshold).astype(int)

class AIDenoising:
    """AI-inspired denoising algorithms"""
    
    @staticmethod
    def non_local_means_denoising(image: np.ndarray, h: float = 10.0, 
                                 search_window: int = 21, patch_size: int = 7) -> np.ndarray:
        """
        Non-local means denoising algorithm
        Compares patches across the image for similarity-based denoising
        """
        try:
            height, width = image.shape
            denoised = np.zeros_like(image, dtype=float)
            
            # Patch radius
            patch_radius = patch_size // 2
            search_radius = search_window // 2
            
            # Precompute patch variance for normalization
            h_squared = h * h
            
            for i in range(patch_radius, height - patch_radius):
                for j in range(patch_radius, width - patch_radius):
                    # Current patch
                    current_patch = image[i-patch_radius:i+patch_radius+1, 
                                        j-patch_radius:j+patch_radius+1]
                    
                    weights_sum = 0.0
                    weighted_sum = 0.0
                    
                    # Search window
                    for ki in range(max(patch_radius, i-search_radius), 
                                  min(height-patch_radius, i+search_radius+1)):
                        for kj in range(max(patch_radius, j-search_radius), 
                                      min(width-patch_radius, j+search_radius+1)):
                            
                            # Comparison patch
                            comp_patch = image[ki-patch_radius:ki+patch_radius+1, 
                                             kj-patch_radius:kj+patch_radius+1]
                            
                            # Compute patch distance
                            patch_diff = current_patch - comp_patch
                            distance = np.sum(patch_diff * patch_diff)
                            
                            # Compute weight
                            weight = np.exp(-max(distance - 2 * h_squared, 0.0) / h_squared)
                            
                            weights_sum += weight
                            weighted_sum += weight * image[ki, kj]
                    
                    # Normalized weighted average
                    if weights_sum > 0:
                        denoised[i, j] = weighted_sum / weights_sum
                    else:
                        denoised[i, j] = image[i, j]
            
            # Handle borders by copying original values
            denoised[:patch_radius, :] = image[:patch_radius, :]
            denoised[-patch_radius:, :] = image[-patch_radius:, :]
            denoised[:, :patch_radius] = image[:, :patch_radius]
            denoised[:, -patch_radius:] = image[:, -patch_radius:]
            
            return denoised
            
        except Exception as e:
            print(f"Non-local means error: {e}")
            # Fallback to simple Gaussian
            from scipy import ndimage
            return ndimage.gaussian_filter(image, sigma=1.0)
    
    @staticmethod
    def bilateral_filter_denoising(image: np.ndarray, sigma_spatial: float = 5.0, 
                                  sigma_intensity: float = 20.0, 
                                  window_size: int = 5) -> np.ndarray:
        """
        Bilateral filtering for edge-preserving denoising
        """
        try:
            height, width = image.shape
            denoised = np.zeros_like(image, dtype=float)
            
            radius = window_size // 2
            
            # Precompute spatial weights
            spatial_weights = np.zeros((window_size, window_size))
            for i in range(window_size):
                for j in range(window_size):
                    di, dj = i - radius, j - radius
                    spatial_weights[i, j] = np.exp(-(di*di + dj*dj) / (2 * sigma_spatial**2))
            
            for i in range(radius, height - radius):
                for j in range(radius, width - radius):
                    center_intensity = image[i, j]
                    
                    weight_sum = 0.0
                    weighted_sum = 0.0
                    
                    for ki in range(i - radius, i + radius + 1):
                        for kj in range(j - radius, j + radius + 1):
                            neighbor_intensity = image[ki, kj]
                            
                            # Intensity weight
                            intensity_diff = center_intensity - neighbor_intensity
                            intensity_weight = np.exp(-(intensity_diff**2) / (2 * sigma_intensity**2))
                            
                            # Combined weight
                            weight = (spatial_weights[ki - i + radius, kj - j + radius] * 
                                    intensity_weight)
                            
                            weight_sum += weight
                            weighted_sum += weight * neighbor_intensity
                    
                    if weight_sum > 0:
                        denoised[i, j] = weighted_sum / weight_sum
                    else:
                        denoised[i, j] = center_intensity
            
            # Handle borders
            denoised[:radius, :] = image[:radius, :]
            denoised[-radius:, :] = image[-radius:, :]
            denoised[:, :radius] = image[:, :radius]
            denoised[:, -radius:] = image[:, -radius:]
            
            return denoised
            
        except Exception as e:
            print(f"Bilateral filter error: {e}")
            # Fallback to mean filter
            from scipy import ndimage
            return ndimage.uniform_filter(image, size=3)
    
    @staticmethod
    def wiener_filter_denoising(image: np.ndarray, noise_variance: float = None) -> np.ndarray:
        """
        Wiener filter denoising in frequency domain
        """
        try:
            # Convert to frequency domain
            f_image = np.fft.fft2(image)
            
            # Estimate noise variance if not provided
            if noise_variance is None:
                # Use Laplacian operator to estimate noise
                laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                from scipy import ndimage
                convolved = ndimage.convolve(image, laplacian)
                noise_variance = np.var(convolved) * 0.5
            
            # Power spectral density
            psd = np.abs(f_image)**2
            
            # Wiener filter
            wiener = psd / (psd + noise_variance)
            
            # Apply filter and convert back
            filtered = f_image * wiener
            denoised = np.real(np.fft.ifft2(filtered))
            
            return denoised
            
        except Exception as e:
            print(f"Wiener filter error: {e}")
            return image

class FourDViewer:
    """4D image viewing and analysis capabilities"""
    
    def __init__(self):
        self.current_volume = None
        self.current_timepoint = 0
        self.current_z_slice = 0
        self.volume_metadata = {}
    
    def load_4d_data(self, image_data: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Load 4D image data (T, Z, Y, X) or (Z, Y, X) for 3D
        """
        try:
            if len(image_data.shape) == 3:
                # Add time dimension for 3D data
                self.current_volume = image_data[np.newaxis, ...]
            elif len(image_data.shape) == 4:
                self.current_volume = image_data
            else:
                # Convert 2D to 4D
                self.current_volume = image_data[np.newaxis, np.newaxis, ...]
            
            self.volume_metadata = metadata or {}
            self.current_timepoint = 0
            self.current_z_slice = 0
            
            return True
            
        except Exception as e:
            print(f"Error loading 4D data: {e}")
            return False
    
    def get_current_slice(self) -> np.ndarray:
        """Get current 2D slice"""
        if self.current_volume is None:
            return np.zeros((100, 100))
        
        try:
            t, z, y, x = self.current_volume.shape
            t_idx = min(self.current_timepoint, t - 1)
            z_idx = min(self.current_z_slice, z - 1)
            
            return self.current_volume[t_idx, z_idx, :, :]
            
        except Exception as e:
            print(f"Error getting slice: {e}")
            return np.zeros((100, 100))
    
    def navigate_time(self, timepoint: int) -> np.ndarray:
        """Navigate to specific timepoint"""
        if self.current_volume is None:
            return np.zeros((100, 100))
        
        max_t = self.current_volume.shape[0]
        self.current_timepoint = max(0, min(timepoint, max_t - 1))
        return self.get_current_slice()
    
    def navigate_z(self, z_slice: int) -> np.ndarray:
        """Navigate to specific Z slice"""
        if self.current_volume is None:
            return np.zeros((100, 100))
        
        max_z = self.current_volume.shape[1]
        self.current_z_slice = max(0, min(z_slice, max_z - 1))
        return self.get_current_slice()
    
    def get_max_projection(self, axis: str = 'z') -> np.ndarray:
        """Get maximum intensity projection"""
        if self.current_volume is None:
            return np.zeros((100, 100))
        
        try:
            current_3d = self.current_volume[self.current_timepoint]
            
            if axis.lower() == 'z':
                return np.max(current_3d, axis=0)
            elif axis.lower() == 'y':
                return np.max(current_3d, axis=1)
            elif axis.lower() == 'x':
                return np.max(current_3d, axis=2)
            else:
                return np.max(current_3d, axis=0)
                
        except Exception as e:
            print(f"Error in max projection: {e}")
            return np.zeros((100, 100))
    
    def get_volume_info(self) -> Dict[str, Any]:
        """Get information about current volume"""
        if self.current_volume is None:
            return {}
        
        t, z, y, x = self.current_volume.shape
        
        return {
            'dimensions': {'T': t, 'Z': z, 'Y': y, 'X': x},
            'current_timepoint': self.current_timepoint,
            'current_z_slice': self.current_z_slice,
            'data_type': str(self.current_volume.dtype),
            'metadata': self.volume_metadata
        }
    
    def create_time_series_projection(self) -> np.ndarray:
        """Create time series maximum projection"""
        if self.current_volume is None:
            return np.zeros((100, 100))
        
        try:
            # Max projection across time
            return np.max(self.current_volume, axis=0)
            
        except Exception as e:
            print(f"Error in time series projection: {e}")
            return np.zeros((100, 100))

class AdvancedAnalyzer:
    """Advanced analysis combining all enhanced features"""
    
    def __init__(self):
        self.segmentation = AdvancedSegmentation()
        self.denoising = AIDenoising()
        self.viewer_4d = FourDViewer()
    
    def process_with_denoising(self, image: np.ndarray, method: str = 'bilateral', 
                             **params) -> np.ndarray:
        """Apply AI denoising before processing"""
        try:
            if method == 'bilateral':
                return self.denoising.bilateral_filter_denoising(image, **params)
            elif method == 'non_local_means':
                return self.denoising.non_local_means_denoising(image, **params)
            elif method == 'wiener':
                return self.denoising.wiener_filter_denoising(image, **params)
            else:
                return image
                
        except Exception as e:
            print(f"Denoising error: {e}")
            return image
    
    def advanced_segment(self, image: np.ndarray, method: str = 'morphological_snakes', 
                        **params) -> np.ndarray:
        """Apply advanced segmentation methods"""
        try:
            if method == 'morphological_snakes':
                return self.segmentation.morphological_snakes(image, **params)
            elif method == 'region_growing':
                return self.segmentation.region_growing(image, **params)
            elif method == 'adaptive_threshold':
                return self.segmentation.adaptive_threshold_segmentation(image, **params)
            else:
                # Fallback to simple threshold
                threshold = np.mean(image)
                return (image > threshold).astype(int)
                
        except Exception as e:
            print(f"Advanced segmentation error: {e}")
            return np.zeros_like(image, dtype=int)
    
    def analyze_4d_volume(self, volume_data: np.ndarray, 
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive 4D volume analysis"""
        try:
            # Load into 4D viewer
            self.viewer_4d.load_4d_data(volume_data, metadata)
            
            # Get volume info
            info = self.viewer_4d.get_volume_info()
            
            # Analyze each timepoint
            timepoint_analysis = []
            t_dim = info['dimensions']['T']
            
            for t in range(min(t_dim, 10)):  # Limit to first 10 timepoints for performance
                slice_2d = self.viewer_4d.navigate_time(t)
                
                # Basic analysis
                timepoint_stats = {
                    'timepoint': t,
                    'mean_intensity': float(np.mean(slice_2d)),
                    'max_intensity': float(np.max(slice_2d)),
                    'std_intensity': float(np.std(slice_2d))
                }
                
                timepoint_analysis.append(timepoint_stats)
            
            # Get projections
            projections = {
                'max_z': self.viewer_4d.get_max_projection('z'),
                'max_y': self.viewer_4d.get_max_projection('y'),
                'max_x': self.viewer_4d.get_max_projection('x')
            }
            
            return {
                'volume_info': info,
                'timepoint_analysis': timepoint_analysis,
                'projections_available': True,
                'analysis_complete': True
            }
            
        except Exception as e:
            print(f"4D analysis error: {e}")
            return {'error': str(e), 'analysis_complete': False}

# Global instance for web interface
advanced_analyzer = AdvancedAnalyzer()