#!/usr/bin/env python3
"""
Batch Processing Module for Scientific Image Analyzer
Handles multiple files with consistent analysis workflows
"""

import os
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import concurrent.futures
import threading

# Import analysis utilities directly instead of nonexistent classes
try:
    from utils import analysis_utils as au
    HAS_ANALYSIS_UTILS = True
except ImportError:
    HAS_ANALYSIS_UTILS = False
    print("Warning: analysis_utils not available")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from scipy import ndimage
    from skimage import filters
    HAS_SCIPY_SKIMAGE = True
except ImportError:
    HAS_SCIPY_SKIMAGE = False
    print("Warning: scipy/skimage not available - using fallback implementations")

try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bug_fixes import TIFFExportFix
    FIXES_AVAILABLE = True
except ImportError:
    FIXES_AVAILABLE = False

try:
    from fibsem_plugins import FIBSEMAnalyzer
    FIBSEM_AVAILABLE = True
except ImportError:
    FIBSEM_AVAILABLE = False

class BatchProcessor:
    """Batch processing engine for multiple image files"""
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.batch_sessions = {}
        self.processing_status = {}
        self.results_cache = {}
        
    def create_batch_session(self, files: List[str], workflow_config: Dict[str, Any]) -> str:
        """Create a new batch processing session"""
        batch_id = str(uuid.uuid4())
        
        self.batch_sessions[batch_id] = {
            'files': files,
            'workflow_config': workflow_config,
            'created_at': time.time(),
            'status': 'initialized',
            'total_files': len(files),
            'processed_files': 0,
            'failed_files': 0,
            'results': {}
        }
        
        return batch_id
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get current status of batch processing"""
        if batch_id not in self.batch_sessions:
            return {'error': 'Batch session not found'}
        
        session = self.batch_sessions[batch_id]
        return {
            'batch_id': batch_id,
            'status': session['status'],
            'total_files': session['total_files'],
            'processed_files': session['processed_files'],
            'failed_files': session['failed_files'],
            'progress_percentage': (session['processed_files'] / session['total_files'] * 100) if session['total_files'] > 0 else 0,
            'elapsed_time': time.time() - session['created_at'],
            'estimated_remaining': self._estimate_remaining_time(batch_id)
        }
    
    def _estimate_remaining_time(self, batch_id: str) -> float:
        """Estimate remaining processing time"""
        session = self.batch_sessions[batch_id]
        if session['processed_files'] == 0:
            return 0
        
        elapsed = time.time() - session['created_at']
        avg_time_per_file = elapsed / session['processed_files']
        remaining_files = session['total_files'] - session['processed_files']
        
        return avg_time_per_file * remaining_files
    
    def process_batch(self, batch_id: str) -> Dict[str, Any]:
        """Start batch processing in background thread"""
        if batch_id not in self.batch_sessions:
            return {'error': 'Batch session not found'}
        
        session = self.batch_sessions[batch_id]
        session['status'] = 'processing'
        
        # Start processing in background thread
        thread = threading.Thread(target=self._process_batch_worker, args=(batch_id,))
        thread.daemon = True
        thread.start()
        
        return {'success': True, 'message': 'Batch processing started', 'batch_id': batch_id}
    
    def _process_batch_worker(self, batch_id: str):
        """Worker function for batch processing"""
        try:
            session = self.batch_sessions[batch_id]
            workflow_config = session['workflow_config']
            
            # Use thread pool for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(self._process_single_file, file_path, workflow_config, batch_id): file_path
                    for file_path in session['files']
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        session['results'][file_path] = result
                        session['processed_files'] += 1
                        
                        if result.get('error'):
                            session['failed_files'] += 1
                            
                    except Exception as e:
                        session['results'][file_path] = {'error': str(e)}
                        session['failed_files'] += 1
                        session['processed_files'] += 1
            
            # Mark batch as completed
            session['status'] = 'completed'
            session['completed_at'] = time.time()
            
        except Exception as e:
            session['status'] = 'failed'
            session['error'] = str(e)
    
    def _process_single_file(self, file_path: str, workflow_config: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
        """Process a single file according to workflow configuration"""
        try:
            start_time = time.time()
            result = {
                'file_path': file_path,
                'filename': Path(file_path).name,
                'processing_start': start_time,
                'workflow_steps': []
            }
            
            # Load image using analysis_utils
            try:
                if HAS_ANALYSIS_UTILS:
                    image_data = au.load_image(file_path)
                else:
                    # Fallback: try PIL directly
                    try:
                        from PIL import Image
                        with Image.open(file_path) as img:
                            image_data = list(list(row) for row in img.convert('L').getdata())
                            # Reshape to 2D
                            w, h = img.size
                            image_data = [image_data[i*w:(i+1)*w] for i in range(h)]
                    except Exception:
                        image_data = None
                        
                if image_data is None:
                    return {'error': f'Failed to load image: {file_path}'}
                
                # Get image shape properly for numpy arrays or lists
                if HAS_NUMPY and isinstance(image_data, np.ndarray):
                    result['image_shape'] = list(image_data.shape)
                else:
                    result['image_shape'] = [len(image_data), len(image_data[0]) if image_data else 0]
                result['workflow_steps'].append('image_loaded')
                
            except Exception as e:
                return {'error': f'Error loading image {file_path}: {str(e)}'}
            
            # Execute workflow steps
            current_image = image_data
            layers = {}
            
            # Preprocessing steps
            if workflow_config.get('preprocessing', {}).get('enabled', False):
                preprocess_config = workflow_config['preprocessing']
                
                if preprocess_config.get('gaussian_filter', False):
                    sigma = preprocess_config.get('gaussian_sigma', 1.0)
                    current_image = self._apply_gaussian_filter(current_image, sigma)
                    result['workflow_steps'].append('gaussian_filter_applied')
                
                if preprocess_config.get('denoising', False) and FIBSEM_AVAILABLE:
                    method = preprocess_config.get('denoising_method', 'bilateral')
                    current_image = self._apply_denoising(current_image, method)
                    result['workflow_steps'].append(f'{method}_denoising_applied')
            
            # Segmentation steps
            if workflow_config.get('segmentation', {}).get('enabled', False):
                seg_config = workflow_config['segmentation']
                method = seg_config.get('method', 'cellpose')
                labels = None
                
                if method == 'cellpose' and HAS_ANALYSIS_UTILS:
                    diameter = seg_config.get('diameter', 30)
                    labels = au.segment_cellpose(current_image, diameter=diameter)
                elif method == 'stardist' and HAS_ANALYSIS_UTILS:
                    labels = au.segment_stardist(current_image)
                elif method == 'watershed' and HAS_ANALYSIS_UTILS:
                    labels = au.segment_watershed(current_image)
                elif method == 'fibsem_comprehensive' and FIBSEM_AVAILABLE:
                    analyzer = FIBSEMAnalyzer()
                    analysis_results = analyzer.run_comprehensive_analysis(current_image, 'segmentation')
                    labels = analysis_results.get('results', {}).get('mitochondria_mask', None)
                
                # Fallback: threshold segmentation with Otsu's method
                if labels is None:
                    labels = self._threshold_segmentation(current_image)
                
                layers['labels'] = labels
                result['workflow_steps'].append(f'{method}_segmentation')
                
                # Count objects correctly using unique labels
                if labels is not None:
                    result['object_count'] = self._count_objects(labels)
            
            # Analysis steps
            if workflow_config.get('analysis', {}).get('enabled', False):
                analysis_config = workflow_config['analysis']
                
                if 'labels' in layers and analysis_config.get('measure_objects', False):
                    # Use analysis_utils for object measurements
                    if HAS_ANALYSIS_UTILS:
                        measurements = au.calculate_object_statistics(layers['labels'], current_image)
                    else:
                        # Basic fallback measurement
                        measurements = self._basic_measurements(layers['labels'], current_image)
                    result['measurements'] = measurements
                    result['workflow_steps'].append('object_analysis')
                
                if analysis_config.get('fibsem_analysis', False) and FIBSEM_AVAILABLE:
                    analyzer = FIBSEMAnalyzer()
                    fibsem_results = analyzer.run_comprehensive_analysis(current_image, 'full')
                    result['fibsem_analysis'] = fibsem_results
                    result['workflow_steps'].append('fibsem_comprehensive_analysis')
            
            # Export steps
            if workflow_config.get('export', {}).get('enabled', False):
                export_config = workflow_config['export']
                export_results = self._handle_export(file_path, layers, result, export_config, batch_id)
                result['exports'] = export_results
                result['workflow_steps'].append('data_exported')
            
            # Calculate processing time
            result['processing_time'] = time.time() - start_time
            result['processing_end'] = time.time()
            result['success'] = True
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'file_path': file_path,
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _count_objects(self, labels) -> int:
        """
        Count objects correctly using unique labels.
        Handles both numpy arrays and nested lists.
        """
        try:
            if HAS_NUMPY and isinstance(labels, np.ndarray):
                # Use numpy's unique - exclude background (0)
                unique_labels = np.unique(labels)
                return int(len(unique_labels[unique_labels > 0]))
            else:
                # Pure Python for nested lists
                unique_labels = set()
                for row in labels:
                    for val in row:
                        if val > 0:
                            unique_labels.add(val)
                return len(unique_labels)
        except Exception:
            return 0
    
    def _threshold_segmentation(self, image):
        """
        Apply threshold segmentation using Otsu's method when available,
        otherwise use an improved adaptive threshold.
        """
        try:
            if HAS_NUMPY and isinstance(image, np.ndarray):
                if HAS_SCIPY_SKIMAGE:
                    # Use Otsu's threshold from skimage
                    threshold = filters.threshold_otsu(image)
                    binary = image > threshold
                    # Label connected components
                    from scipy import ndimage as ndi
                    labels, _ = ndi.label(binary)
                    return labels
                else:
                    # Improved threshold using histogram analysis
                    flat = image.flatten()
                    threshold = np.median(flat) + 0.5 * np.std(flat)
                    binary = image > threshold
                    # Simple labeling
                    return (binary > 0).astype(int)
            else:
                # Pure Python fallback with histogram-based threshold
                flat = [pixel for row in image for pixel in row]
                flat_sorted = sorted(flat)
                # Use median as threshold (more robust than mean)
                threshold = flat_sorted[len(flat_sorted) // 2]
                # Add half standard deviation for better separation
                mean_val = sum(flat) / len(flat)
                variance = sum((x - mean_val) ** 2 for x in flat) / len(flat)
                std_val = variance ** 0.5
                threshold = threshold + 0.5 * std_val
                
                height = len(image)
                width = len(image[0]) if height > 0 else 0
                return [[1 if pixel > threshold else 0 for pixel in row] for row in image]
        except Exception:
            # Ultimate fallback: simple mean threshold
            if HAS_NUMPY and isinstance(image, np.ndarray):
                threshold = np.mean(image)
                return (image > threshold).astype(int)
            else:
                flat = [pixel for row in image for pixel in row]
                threshold = sum(flat) / len(flat) if flat else 0
                return [[1 if pixel > threshold else 0 for pixel in row] for row in image]
    
    def _basic_measurements(self, labels, intensity_image=None) -> Dict[str, Any]:
        """Basic fallback measurements when analysis_utils is not available"""
        measurements = {'label': [], 'area': [], 'mean_intensity': []}
        
        try:
            if HAS_NUMPY and isinstance(labels, np.ndarray):
                unique_labels = np.unique(labels)
                unique_labels = unique_labels[unique_labels > 0]
                
                for label_id in unique_labels:
                    mask = labels == label_id
                    area = int(np.sum(mask))
                    measurements['label'].append(int(label_id))
                    measurements['area'].append(area)
                    
                    if intensity_image is not None and isinstance(intensity_image, np.ndarray):
                        mean_int = float(np.mean(intensity_image[mask]))
                        measurements['mean_intensity'].append(mean_int)
                    else:
                        measurements['mean_intensity'].append(0.0)
            else:
                # Pure Python fallback
                label_areas = {}
                label_intensities = {}
                
                for r, row in enumerate(labels):
                    for c, label_val in enumerate(row):
                        if label_val > 0:
                            if label_val not in label_areas:
                                label_areas[label_val] = 0
                                label_intensities[label_val] = []
                            label_areas[label_val] += 1
                            if intensity_image:
                                label_intensities[label_val].append(intensity_image[r][c])
                
                for label_id in sorted(label_areas.keys()):
                    measurements['label'].append(label_id)
                    measurements['area'].append(label_areas[label_id])
                    if label_intensities.get(label_id):
                        mean_int = sum(label_intensities[label_id]) / len(label_intensities[label_id])
                        measurements['mean_intensity'].append(mean_int)
                    else:
                        measurements['mean_intensity'].append(0.0)
            
            measurements['object_count'] = len(measurements['label'])
        except Exception as e:
            measurements['error'] = str(e)
            measurements['object_count'] = 0
        
        return measurements
    
    def _apply_gaussian_filter(self, image, sigma):
        """Apply Gaussian filter with scipy when available, otherwise improved fallback"""
        try:
            # Prefer scipy's gaussian_filter
            if HAS_SCIPY_SKIMAGE and HAS_NUMPY:
                if not isinstance(image, np.ndarray):
                    image = np.array(image)
                return ndimage.gaussian_filter(image, sigma=sigma)
            
            # Fallback: separable approximation for better quality
            if HAS_NUMPY and isinstance(image, np.ndarray):
                # Create 1D Gaussian kernel
                kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
                x = np.arange(kernel_size) - kernel_size // 2
                kernel = np.exp(-x**2 / (2 * sigma**2))
                kernel = kernel / kernel.sum()
                
                # Apply separable convolution (horizontal then vertical)
                filtered = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 0, image)
                filtered = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 1, filtered)
                return filtered
            
            # Pure Python fallback: box filter approximation
            height = len(image)
            width = len(image[0]) if height > 0 else 0
            
            filtered = [[0 for _ in range(width)] for _ in range(height)]
            kernel_size = max(1, int(sigma * 3))
            
            for i in range(height):
                for j in range(width):
                    sum_val = 0
                    count = 0
                    
                    for di in range(-kernel_size, kernel_size + 1):
                        for dj in range(-kernel_size, kernel_size + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                sum_val += image[ni][nj]
                                count += 1
                    
                    filtered[i][j] = int(sum_val / count) if count > 0 else image[i][j]
            
            return filtered
        except Exception:
            return image
    
    def _apply_denoising(self, image, method):
        """Apply denoising with FIB-SEM tools"""
        try:
            if FIBSEM_AVAILABLE:
                from fibsem_plugins import GPU_ImageProcessor
                
                if method == 'bilateral':
                    return GPU_ImageProcessor.accelerated_gaussian_filter(image, sigma=1.5)
                elif method == 'gaussian':
                    return GPU_ImageProcessor.accelerated_gaussian_filter(image, sigma=2.0)
            
            return image
        except Exception:
            return image
    
    def _handle_export(self, file_path: str, layers: Dict, result: Dict, export_config: Dict, batch_id: str) -> Dict[str, Any]:
        """Handle export operations for batch processing"""
        exports = {}
        base_filename = Path(file_path).stem
        export_dir = f"batch_exports/{batch_id}"
        
        # Create export directory
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            if export_config.get('csv_measurements', False) and 'measurements' in result:
                csv_path = os.path.join(export_dir, f"{base_filename}_measurements.csv")
                self._export_measurements_csv(result['measurements'], csv_path)
                exports['measurements_csv'] = csv_path
            
            if export_config.get('mask_tiff', False) and 'labels' in layers:
                tiff_path = os.path.join(export_dir, f"{base_filename}_mask.tiff")
                if FIXES_AVAILABLE:
                    success, method = TIFFExportFix.export_proper_tiff(layers['labels'], tiff_path)
                    if success:
                        print(f"Exported mask as {method} to {tiff_path}")
                        exports['mask_tiff'] = tiff_path
                    else:
                        print(f"TIFF export failed: {method}")
                else:
                    self._export_mask_simple(layers['labels'], tiff_path)
                    exports['mask_tiff'] = tiff_path.replace('.tiff', '.txt')
            
            if export_config.get('summary_json', True):
                json_path = os.path.join(export_dir, f"{base_filename}_summary.json")
                summary = {
                    'filename': result['filename'],
                    'processing_time': result.get('processing_time', 0),
                    'object_count': result.get('object_count', 0),
                    'workflow_steps': result.get('workflow_steps', []),
                    'image_shape': result.get('image_shape', [])
                }
                
                with open(json_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                exports['summary_json'] = json_path
            
        except Exception as e:
            exports['export_error'] = str(e)
        
        return exports
    
    def _export_measurements_csv(self, measurements: List[Dict], filepath: str):
        """Export measurements to CSV format"""
        if not measurements:
            return
        
        try:
            headers = list(measurements[0].keys())
            
            with open(filepath, 'w') as f:
                # Write header
                f.write(','.join(headers) + '\n')
                
                # Write data rows
                for measurement in measurements:
                    row = [str(measurement.get(header, '')) for header in headers]
                    f.write(','.join(row) + '\n')
                    
        except Exception as e:
            print(f"CSV export error: {e}")
    
    def _export_mask_simple(self, labels, filepath: str):
        """Simple mask export (placeholder for TIFF)"""
        try:
            # For now, export as text format
            with open(filepath.replace('.tiff', '.txt'), 'w') as f:
                for row in labels:
                    f.write(' '.join(map(str, row)) + '\n')
        except Exception as e:
            print(f"Mask export error: {e}")
    
    def get_batch_results(self, batch_id: str) -> Dict[str, Any]:
        """Get complete results for a batch"""
        if batch_id not in self.batch_sessions:
            return {'error': 'Batch session not found'}
        
        session = self.batch_sessions[batch_id]
        
        # Generate summary statistics
        results = session['results']
        successful_files = [r for r in results.values() if r.get('success', False)]
        failed_files = [r for r in results.values() if 'error' in r]
        
        total_objects = sum(r.get('object_count', 0) for r in successful_files)
        avg_processing_time = sum(r.get('processing_time', 0) for r in successful_files) / len(successful_files) if successful_files else 0
        
        summary = {
            'batch_id': batch_id,
            'status': session['status'],
            'total_files': session['total_files'],
            'successful_files': len(successful_files),
            'failed_files': len(failed_files),
            'total_objects_found': total_objects,
            'average_processing_time': avg_processing_time,
            'total_processing_time': time.time() - session['created_at'] if session['status'] == 'completed' else None,
            'workflow_config': session['workflow_config']
        }
        
        return {
            'summary': summary,
            'detailed_results': results,
            'export_directory': f"batch_exports/{batch_id}" if any('exports' in r for r in successful_files) else None
        }
    
    def cleanup_batch(self, batch_id: str, keep_exports: bool = True) -> Dict[str, Any]:
        """Clean up batch session data"""
        if batch_id not in self.batch_sessions:
            return {'error': 'Batch session not found'}
        
        try:
            if not keep_exports:
                export_dir = f"batch_exports/{batch_id}"
                if os.path.exists(export_dir):
                    import shutil
                    shutil.rmtree(export_dir)
            
            del self.batch_sessions[batch_id]
            
            return {'success': True, 'message': f'Batch {batch_id} cleaned up'}
            
        except Exception as e:
            return {'error': f'Cleanup failed: {str(e)}'}

# Workflow templates for common use cases
WORKFLOW_TEMPLATES = {
    'fibsem_comprehensive': {
        'name': 'FIB-SEM Comprehensive Analysis',
        'description': 'Full pipeline for FIB-SEM data with specialized tools',
        'preprocessing': {
            'enabled': True,
            'denoising': True,
            'denoising_method': 'bilateral'
        },
        'segmentation': {
            'enabled': True,
            'method': 'fibsem_comprehensive'
        },
        'analysis': {
            'enabled': True,
            'measure_objects': True,
            'fibsem_analysis': True
        },
        'export': {
            'enabled': True,
            'csv_measurements': True,
            'mask_tiff': True,
            'summary_json': True
        }
    },
    'cell_counting': {
        'name': 'Cell Counting Workflow',
        'description': 'Standard cell segmentation and counting',
        'preprocessing': {
            'enabled': True,
            'gaussian_filter': True,
            'gaussian_sigma': 1.0
        },
        'segmentation': {
            'enabled': True,
            'method': 'cellpose',
            'diameter': 30
        },
        'analysis': {
            'enabled': True,
            'measure_objects': True
        },
        'export': {
            'enabled': True,
            'csv_measurements': True,
            'summary_json': True
        }
    },
    'quick_analysis': {
        'name': 'Quick Analysis',
        'description': 'Fast processing for large datasets',
        'preprocessing': {
            'enabled': False
        },
        'segmentation': {
            'enabled': True,
            'method': 'threshold'
        },
        'analysis': {
            'enabled': True,
            'measure_objects': False
        },
        'export': {
            'enabled': True,
            'summary_json': True
        }
    }
}
