"""
Analysis Utility Functions for Scientific Image Analyzer
Provides statistical analysis and colocalization utilities
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from scipy import ndimage, stats
from skimage import measure, filters, morphology
import pandas as pd

def calculate_object_statistics(labeled_image: np.ndarray, 
                               intensity_image: Optional[np.ndarray] = None,
                               properties: Optional[List[str]] = None) -> Dict:
    """
    Calculate comprehensive statistics for labeled objects
    
    Args:
        labeled_image: Labeled image with object regions
        intensity_image: Optional intensity image for intensity-based measurements
        properties: List of properties to calculate
        
    Returns:
        Dictionary containing calculated statistics
    """
    if properties is None:
        properties = [
            'label', 'area', 'centroid', 'bbox', 'perimeter',
            'major_axis_length', 'minor_axis_length', 'eccentricity',
            'orientation', 'solidity', 'extent'
        ]
    
    # Add intensity properties if intensity image is provided
    if intensity_image is not None:
        intensity_props = [
            'mean_intensity', 'max_intensity', 'min_intensity',
            'weighted_centroid'
        ]
        properties.extend(intensity_props)
    
    try:
        # Calculate region properties
        if intensity_image is not None:
            regions = measure.regionprops(labeled_image, intensity_image)
        else:
            regions = measure.regionprops(labeled_image)
        
        # Extract statistics
        stats_dict = {}
        
        for prop in properties:
            if hasattr(regions[0], prop):
                values = []
                for region in regions:
                    val = getattr(region, prop)
                    if isinstance(val, (tuple, list, np.ndarray)):
                        # For multi-dimensional properties, store as arrays
                        values.append(val)
                    else:
                        values.append(val)
                stats_dict[prop] = values
        
        # Calculate additional derived statistics
        if 'area' in stats_dict:
            areas = np.array(stats_dict['area'])
            stats_dict.update({
                'total_area': np.sum(areas),
                'mean_area': np.mean(areas),
                'std_area': np.std(areas),
                'min_area': np.min(areas),
                'max_area': np.max(areas),
                'median_area': np.median(areas)
            })
        
        if 'mean_intensity' in stats_dict:
            intensities = np.array(stats_dict['mean_intensity'])
            stats_dict.update({
                'mean_object_intensity': np.mean(intensities),
                'std_object_intensity': np.std(intensities),
                'min_object_intensity': np.min(intensities),
                'max_object_intensity': np.max(intensities)
            })
        
        # Object count
        stats_dict['object_count'] = len(regions)
        
        return stats_dict
        
    except Exception as e:
        return {'error': str(e), 'object_count': 0}

def calculate_colocalization_coefficients(image1: np.ndarray, 
                                        image2: np.ndarray) -> Dict[str, float]:
    """
    Calculate various colocalization coefficients
    
    Args:
        image1: First channel image
        image2: Second channel image
        
    Returns:
        Dictionary of colocalization coefficients
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape")
    
    # Flatten images for calculations
    ch1 = image1.flatten().astype(np.float64)
    ch2 = image2.flatten().astype(np.float64)
    
    # Remove zero pixels for some calculations
    mask = (ch1 > 0) & (ch2 > 0)
    ch1_nz = ch1[mask]
    ch2_nz = ch2[mask]
    
    coefficients = {}
    
    try:
        # Pearson correlation coefficient
        if len(ch1_nz) > 1:
            pearson_r, pearson_p = stats.pearsonr(ch1_nz, ch2_nz)
            coefficients['pearson_correlation'] = pearson_r
            coefficients['pearson_p_value'] = pearson_p
        else:
            coefficients['pearson_correlation'] = 0.0
            coefficients['pearson_p_value'] = 1.0
        
        # Spearman correlation coefficient
        if len(ch1_nz) > 1:
            spearman_r, spearman_p = stats.spearmanr(ch1_nz, ch2_nz)
            coefficients['spearman_correlation'] = spearman_r
            coefficients['spearman_p_value'] = spearman_p
        else:
            coefficients['spearman_correlation'] = 0.0
            coefficients['spearman_p_value'] = 1.0
        
        # Overlap coefficients (Manders-like)
        sum_ch1 = np.sum(ch1)
        sum_ch2 = np.sum(ch2)
        
        if sum_ch1 > 0 and sum_ch2 > 0:
            # K1: fraction of ch1 intensity that overlaps with ch2
            overlap_ch1 = np.sum(ch1 * (ch2 > 0))
            coefficients['overlap_k1'] = overlap_ch1 / sum_ch1
            
            # K2: fraction of ch2 intensity that overlaps with ch1
            overlap_ch2 = np.sum(ch2 * (ch1 > 0))
            coefficients['overlap_k2'] = overlap_ch2 / sum_ch2
        else:
            coefficients['overlap_k1'] = 0.0
            coefficients['overlap_k2'] = 0.0
        
        # Costes' automatic threshold approach would go here
        # For now, placeholder values
        coefficients['costes_p_value'] = 0.0
        coefficients['costes_correlation_random'] = 0.0
        
    except Exception as e:
        # Return default values on error
        coefficients = {
            'pearson_correlation': 0.0,
            'pearson_p_value': 1.0,
            'spearman_correlation': 0.0,
            'spearman_p_value': 1.0,
            'overlap_k1': 0.0,
            'overlap_k2': 0.0,
            'costes_p_value': 0.0,
            'costes_correlation_random': 0.0,
            'error': str(e)
        }
    
    return coefficients

def manders_coefficients(image1: np.ndarray, image2: np.ndarray,
                        threshold1: float = 0, threshold2: float = 0) -> Tuple[float, float]:
    """
    Calculate Manders colocalization coefficients
    
    Args:
        image1: First channel image
        image2: Second channel image
        threshold1: Threshold for first channel
        threshold2: Threshold for second channel
        
    Returns:
        Tuple of (M1, M2) coefficients
    """
    # Apply thresholds
    mask1 = image1 > threshold1
    mask2 = image2 > threshold2
    
    # Calculate coefficients
    sum1_total = np.sum(image1[mask1])
    sum2_total = np.sum(image2[mask2])
    
    if sum1_total > 0:
        sum1_coloc = np.sum(image1[mask1 & mask2])
        m1 = sum1_coloc / sum1_total
    else:
        m1 = 0.0
    
    if sum2_total > 0:
        sum2_coloc = np.sum(image2[mask1 & mask2])
        m2 = sum2_coloc / sum2_total
    else:
        m2 = 0.0
    
    return m1, m2

def costes_threshold(image1: np.ndarray, image2: np.ndarray) -> Tuple[float, float]:
    """
    Calculate automatic thresholds using Costes method
    
    Args:
        image1: First channel image
        image2: Second channel image
        
    Returns:
        Tuple of threshold values for (image1, image2)
    """
    # Simplified implementation of Costes method
    # This is a basic version - full implementation would be more complex
    
    # Use Otsu's method as approximation
    try:
        thresh1 = filters.threshold_otsu(image1)
        thresh2 = filters.threshold_otsu(image2)
        return float(thresh1), float(thresh2)
    except:
        # Fallback to percentile method
        thresh1 = np.percentile(image1, 75)
        thresh2 = np.percentile(image2, 75)
        return float(thresh1), float(thresh2)

def calculate_intensity_statistics(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive intensity statistics for an image
    
    Args:
        image: Input image array
        mask: Optional binary mask to restrict analysis
        
    Returns:
        Dictionary of intensity statistics
    """
    if mask is not None:
        data = image[mask]
    else:
        data = image.flatten()
    
    if len(data) == 0:
        return {'error': 'No data to analyze'}
    
    stats_dict = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
        'cv': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0,
        'pixel_count': len(data),
        'sum': float(np.sum(data))
    }
    
    return stats_dict

def analyze_spatial_distribution(coordinates: np.ndarray) -> Dict[str, float]:
    """
    Analyze spatial distribution of points/objects
    
    Args:
        coordinates: Array of coordinates (N x 2 or N x 3)
        
    Returns:
        Dictionary of spatial statistics
    """
    if coordinates.shape[0] < 2:
        return {'error': 'Need at least 2 points for spatial analysis'}
    
    from scipy.spatial.distance import pdist, cdist
    
    stats_dict = {}
    
    try:
        # Calculate all pairwise distances
        distances = pdist(coordinates)
        
        stats_dict.update({
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'median_distance': float(np.median(distances))
        })
        
        # Nearest neighbor distances
        distance_matrix = cdist(coordinates, coordinates)
        np.fill_diagonal(distance_matrix, np.inf)
        nn_distances = np.min(distance_matrix, axis=1)
        
        stats_dict.update({
            'mean_nn_distance': float(np.mean(nn_distances)),
            'std_nn_distance': float(np.std(nn_distances)),
            'min_nn_distance': float(np.min(nn_distances)),
            'max_nn_distance': float(np.max(nn_distances))
        })
        
        # Density estimation (points per unit area/volume)
        if coordinates.shape[1] == 2:
            # 2D case
            x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
            y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
            area = x_range * y_range
            if area > 0:
                stats_dict['density'] = len(coordinates) / area
            else:
                stats_dict['density'] = 0
        elif coordinates.shape[1] == 3:
            # 3D case
            x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
            y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
            z_range = np.max(coordinates[:, 2]) - np.min(coordinates[:, 2])
            volume = x_range * y_range * z_range
            if volume > 0:
                stats_dict['density'] = len(coordinates) / volume
            else:
                stats_dict['density'] = 0
        
        # Center of mass
        center_of_mass = np.mean(coordinates, axis=0)
        stats_dict['center_x'] = float(center_of_mass[0])
        stats_dict['center_y'] = float(center_of_mass[1])
        if coordinates.shape[1] == 3:
            stats_dict['center_z'] = float(center_of_mass[2])
        
    except Exception as e:
        stats_dict['error'] = str(e)
    
    return stats_dict

def calculate_surface_statistics(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for 3D surface meshes
    
    Args:
        vertices: Array of vertex coordinates
        faces: Array of face indices
        
    Returns:
        Dictionary of surface statistics
    """
    stats_dict = {}
    
    try:
        # Basic counts
        stats_dict.update({
            'vertex_count': len(vertices),
            'face_count': len(faces),
            'edge_count': len(faces) * 3 // 2  # Approximation for triangular mesh
        })
        
        # Bounding box
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        bbox_size = bbox_max - bbox_min
        
        stats_dict.update({
            'bbox_x': float(bbox_size[0]),
            'bbox_y': float(bbox_size[1]),
            'bbox_z': float(bbox_size[2]),
            'bbox_volume': float(np.prod(bbox_size))
        })
        
        # Center of vertices
        center = np.mean(vertices, axis=0)
        stats_dict.update({
            'center_x': float(center[0]),
            'center_y': float(center[1]),
            'center_z': float(center[2])
        })
        
        # Surface area calculation (for triangular faces)
        if faces.shape[1] == 3:
            total_area = 0
            for face in faces:
                v1, v2, v3 = vertices[face]
                # Calculate triangle area using cross product
                edge1 = v2 - v1
                edge2 = v3 - v1
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                total_area += area
            
            stats_dict['surface_area'] = float(total_area)
        
        # Volume calculation (for closed meshes)
        # This is a simplified calculation
        if faces.shape[1] == 3:
            volume = 0
            for face in faces:
                v1, v2, v3 = vertices[face]
                # Tetrahedron volume from origin
                volume += np.dot(v1, np.cross(v2, v3)) / 6
            
            stats_dict['volume'] = float(abs(volume))
        
    except Exception as e:
        stats_dict['error'] = str(e)
    
    return stats_dict

def create_analysis_report(results: Dict, output_format: str = 'dict') -> Union[Dict, str, pd.DataFrame]:
    """
    Create formatted analysis report from results
    
    Args:
        results: Dictionary of analysis results
        output_format: Output format ('dict', 'text', 'dataframe')
        
    Returns:
        Formatted report in requested format
    """
    if output_format == 'dict':
        return results
    
    elif output_format == 'text':
        report_lines = []
        report_lines.append("SCIENTIFIC IMAGE ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        for section, data in results.items():
            report_lines.append(f"{section.upper().replace('_', ' ')}")
            report_lines.append("-" * len(section))
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, float):
                        report_lines.append(f"  {key}: {value:.4f}")
                    else:
                        report_lines.append(f"  {key}: {value}")
            else:
                report_lines.append(f"  {data}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    elif output_format == 'dataframe':
        # Flatten nested dictionaries for DataFrame
        flat_data = {}
        for section, data in results.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    flat_data[f"{section}_{key}"] = [value]
            else:
                flat_data[section] = [data]
        
        return pd.DataFrame(flat_data)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def batch_analysis(image_list: List[np.ndarray], 
                  analysis_functions: List[callable],
                  function_kwargs: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Perform batch analysis on multiple images
    
    Args:
        image_list: List of images to analyze
        analysis_functions: List of analysis functions to apply
        function_kwargs: Optional list of keyword arguments for each function
        
    Returns:
        List of analysis results for each image
    """
    if function_kwargs is None:
        function_kwargs = [{}] * len(analysis_functions)
    
    results = []
    
    for i, image in enumerate(image_list):
        image_results = {'image_index': i}
        
        for j, func in enumerate(analysis_functions):
            try:
                kwargs = function_kwargs[j] if j < len(function_kwargs) else {}
                result = func(image, **kwargs)
                function_name = func.__name__
                image_results[function_name] = result
            except Exception as e:
                function_name = func.__name__
                image_results[function_name] = {'error': str(e)}
        
        results.append(image_results)
    
    return results

def quality_assessment(image: np.ndarray) -> Dict[str, float]:
    """
    Assess image quality metrics
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    try:
        # Signal-to-noise ratio estimation
        if image.ndim == 2:
            # Use Laplacian variance as sharpness measure
            laplacian_var = ndimage.generic_laplacian(image.astype(float), np.ones((3, 3))).var()
            metrics['sharpness'] = float(laplacian_var)
        
        # Dynamic range
        metrics['dynamic_range'] = float(np.max(image) - np.min(image))
        
        # Contrast (RMS contrast)
        mean_intensity = np.mean(image)
        contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))
        metrics['contrast'] = float(contrast)
        
        # Brightness
        metrics['brightness'] = float(mean_intensity)
        
        # Histogram entropy (measure of information content)
        hist, _ = np.histogram(image, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log2(hist))
        metrics['entropy'] = float(entropy)
        
        # Noise estimation (using high-frequency content)
        if image.ndim >= 2:
            # Simple noise estimate using standard deviation of Laplacian
            laplacian = ndimage.laplace(image.astype(float))
            noise_estimate = np.std(laplacian)
            metrics['noise_estimate'] = float(noise_estimate)
        
    except Exception as e:
        metrics['error'] = str(e)
    
    return metrics
