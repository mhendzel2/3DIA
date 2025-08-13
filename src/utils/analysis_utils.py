"""
Analysis Utility Functions for Scientific Image Analyzer
Provides statistical analysis and colocalization utilities with robust fallbacks.
"""
import math

try:
    import numpy as np
    from scipy import ndimage, stats
    from skimage import measure, morphology, feature, segmentation, filters
    import pandas as pd
    from PIL import Image
    from cellpose import models
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    HAS_SCIENTIFIC_LIBS = True
except ImportError:
    HAS_SCIENTIFIC_LIBS = False
    print("Warning: numpy, scipy, scikit-image, cellpose, or stardist not found. Using pure Python fallbacks.")

def load_image(file_path):
    """Load image and convert to numpy array."""
    if not HAS_SCIENTIFIC_LIBS:
        return None
    try:
        with Image.open(file_path) as img:
            return np.array(img.convert('L')) # Convert to grayscale
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def segment_cellpose(image, diameter=30):
    """
    Segment cells using the real Cellpose model.
    """
    if not HAS_SCIENTIFIC_LIBS:
        print("Cellpose not available.")
        return None
    model = models.Cellpose(model_type='cyto')
    masks, _, _, _ = model.eval(image, diameter=diameter, channels=[0,0])
    return masks

def segment_stardist(image):
    """
    Segment nuclei using the real StarDist model.
    """
    if not HAS_SCIENTIFIC_LIBS:
        print("StarDist not available.")
        return None
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    img_norm = normalize(image, 1, 99.8, axis=(0,1))
    labels, _ = model.predict_instances(img_norm)
    return labels

def segment_watershed(image):
    """
    Segment using a standard watershed algorithm.
    """
    if not HAS_SCIENTIFIC_LIBS:
        print("Watershed not available.")
        return None
    distance = ndi.distance_transform_edt(image)
    coords = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=image)
    return labels

def calculate_object_statistics(labeled_image, intensity_image=None, properties=None):
    """
    Calculate comprehensive statistics for labeled objects.
    Uses scikit-image if available, otherwise provides basic pure Python measurements.
    """
    if HAS_SCIENTIFIC_LIBS:
        try:
            return _calculate_object_statistics_skimage(labeled_image, intensity_image, properties)
        except Exception as e:
            print(f"Scikit-image analysis failed: {e}. Using fallback.")
            return _calculate_object_statistics_fallback(labeled_image, intensity_image)
    else:
        return _calculate_object_statistics_fallback(labeled_image, intensity_image)

def _calculate_object_statistics_skimage(labeled_image, intensity_image=None, properties=None):
    """scikit-image based implementation for object statistics."""
    if properties is None:
        properties = [
            'label', 'area', 'centroid', 'bbox', 'perimeter', 'convex_area',
            'major_axis_length', 'minor_axis_length', 'eccentricity',
            'orientation', 'solidity', 'extent', 'feret_diameter_max'
        ]
    
    # Add intensity properties if intensity image is provided
    if intensity_image is not None:
        intensity_props = [
            'mean_intensity', 'max_intensity', 'min_intensity',
            'weighted_centroid'
        ]
        properties.extend(intensity_props)
    
    # Calculate region properties using skimage
    regions = measure.regionprops_table(
        labeled_image,
        intensity_image=intensity_image,
        properties=properties
    )
    
    df = pd.DataFrame(regions)
    
    # Calculate custom Circularity (requires perimeter and area)
    if 'area' in df and 'perimeter' in df:
        df['circularity'] = df.apply(
            lambda row: (4 * np.pi * row['area']) / (row['perimeter']**2) if row['perimeter'] > 0 else 0,
            axis=1
        )
    
    stats_dict = df.to_dict('list')
    
    # Calculate additional derived statistics
    if 'area' in stats_dict:
        areas = np.array(stats_dict['area'])
        stats_dict.update({
            'total_area': float(np.sum(areas)),
            'mean_area': float(np.mean(areas)),
            'std_area': float(np.std(areas)),
            'min_area': float(np.min(areas)),
            'max_area': float(np.max(areas)),
            'median_area': float(np.median(areas))
        })
    
    if 'mean_intensity' in stats_dict:
        intensities = np.array(stats_dict['mean_intensity'])
        stats_dict.update({
            'mean_object_intensity': float(np.mean(intensities)),
            'std_object_intensity': float(np.std(intensities)),
            'min_object_intensity': float(np.min(intensities)),
            'max_object_intensity': float(np.max(intensities))
        })
    
    # Object count
    stats_dict['object_count'] = len(df)
    
    return stats_dict

def _calculate_object_statistics_fallback(labels, intensity_image=None):
    """Pure Python fallback for basic object measurements."""
    unique_labels = set(pixel for row in labels for pixel in row if pixel > 0)
    results = {'label': [], 'area': [], 'mean_intensity': []}
    
    for label_id in unique_labels:
        area = 0
        intensity_sum = 0
        pixels = []
        for r, row in enumerate(labels):
            for c, pixel_label in enumerate(row):
                if pixel_label == label_id:
                    area += 1
                    if intensity_image:
                        intensity_sum += intensity_image[r][c]
        
        results['label'].append(label_id)
        results['area'].append(area)
        if intensity_image and area > 0:
            results['mean_intensity'].append(intensity_sum / area)
        else:
             results['mean_intensity'].append(0)

    results['object_count'] = len(unique_labels)
    return results

def calculate_colocalization_coefficients(image1, image2, 
                                         threshold1=None,
                                         threshold2=None):
    """
    Calculate various colocalization coefficients.
    Uses scipy if available, otherwise provides basic pure Python measurements.
    """
    if HAS_SCIENTIFIC_LIBS:
        try:
            return _calculate_colocalization_coefficients_scipy(image1, image2, threshold1, threshold2)
        except Exception as e:
            print(f"Scipy colocalization failed: {e}. Using fallback.")
            return _calculate_colocalization_coefficients_fallback(image1, image2)
    else:
        return _calculate_colocalization_coefficients_fallback(image1, image2)

def _calculate_colocalization_coefficients_scipy(image1, image2, threshold1=None, threshold2=None):
    """scipy-based implementation for colocalization."""
    if hasattr(image1, 'shape') and hasattr(image2, 'shape'):
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same shape")
    
    if not isinstance(image1, np.ndarray):
        image1 = np.array(image1)
    if not isinstance(image2, np.ndarray):
        image2 = np.array(image2)
    
    ch1 = image1.flatten().astype(np.float64)
    ch2 = image2.flatten().astype(np.float64)
    
    pearson_r, _ = stats.pearsonr(ch1, ch2)
    
    # Calculate thresholds if not provided
    if threshold1 is None or threshold2 is None:
        thresh1, thresh2 = calculate_automatic_thresholds(image1, image2)
        if threshold1 is None:
            threshold1 = thresh1
        if threshold2 is None:
            threshold2 = thresh2
    
    m1, m2 = manders_coefficients(image1, image2, threshold1, threshold2)

    return {
        'pearson_correlation': float(pearson_r),
        'manders_m1': float(m1),
        'manders_m2': float(m2),
        'threshold1': float(threshold1),
        'threshold2': float(threshold2)
    }

def _calculate_colocalization_coefficients_fallback(image1, image2):
    """Pure Python fallback for colocalization."""
    # Flatten images
    ch1 = [pixel for row in image1 for pixel in row]
    ch2 = [pixel for row in image2 for pixel in row]

    if not ch1 or not ch2: 
        return {'pearson_correlation': 0.0, 'manders_m1': 0.0, 'manders_m2': 0.0}

    # Pearson correlation
    mean1 = sum(ch1) / len(ch1)
    mean2 = sum(ch2) / len(ch2)
    
    numerator = sum((ch1[i] - mean1) * (ch2[i] - mean2) for i in range(len(ch1)))
    denom1 = sum((p - mean1)**2 for p in ch1)
    denom2 = sum((p - mean2)**2 for p in ch2)
    
    pearson_r = numerator / math.sqrt(denom1 * denom2) if denom1 > 0 and denom2 > 0 else 0
    
    return {'pearson_correlation': pearson_r, 'manders_m1': 0.0, 'manders_m2': 0.0}

def calculate_automatic_thresholds(image1, image2):
    """
    Calculate automatic thresholds for colocalization analysis
    """
    
    # Use Otsu's method as approximation
    try:
        if HAS_SCIENTIFIC_LIBS:
            from skimage import filters
            thresh1 = filters.threshold_otsu(image1)
            thresh2 = filters.threshold_otsu(image2)
            return float(thresh1), float(thresh2)
        else:
            raise ImportError("Scientific libraries not available")
    except ImportError:
        # Fallback to percentile-based thresholding
        if HAS_SCIENTIFIC_LIBS:
            thresh1 = np.percentile(image1, 75)
            thresh2 = np.percentile(image2, 75)
        else:
            # Pure Python percentile calculation
            flat1 = [pixel for row in image1 for pixel in row]
            flat2 = [pixel for row in image2 for pixel in row]
            flat1.sort()
            flat2.sort()
            thresh1 = flat1[int(len(flat1) * 0.75)] if flat1 else 0
            thresh2 = flat2[int(len(flat2) * 0.75)] if flat2 else 0
        return float(thresh1), float(thresh2)

def manders_coefficients(image1, image2, threshold1=0, threshold2=0):
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

def costes_threshold(image1, image2):
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
        from skimage import filters
        thresh1 = filters.threshold_otsu(image1)
        thresh2 = filters.threshold_otsu(image2)
        return float(thresh1), float(thresh2)
    except:
        # Fallback to percentile method
        thresh1 = np.percentile(image1, 75)
        thresh2 = np.percentile(image2, 75)
        return float(thresh1), float(thresh2)

def calculate_intensity_statistics(image, mask=None):
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

def analyze_spatial_distribution(coordinates):
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

def calculate_surface_statistics(vertices, faces):
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

def create_analysis_report(results, output_format='dict'):
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

def batch_analysis(image_list, analysis_functions, function_kwargs=None):
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
                pass
        
        results.append(image_results)
    
    return results

def quality_assessment(image):
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
            laplacian = ndimage.laplace(image.astype(float))
            laplacian_var = laplacian.var()
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
        hist_safe = hist.astype(float)
        entropy = float(-np.sum(hist_safe * np.log2(np.maximum(hist_safe, 1e-10))))
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

def four_param_logistic(x, A, B, C, D):
    """4PL logistic function for dose-response curves."""
    return D + (A - D) / (1 + (x / C)**B)

def fit_dose_response(concentrations, responses):
    """
    Fits a 4PL model to dose-response data to find the IC50.
    
    Args:
        concentrations (list): List of drug concentrations.
        responses (list): List of corresponding cellular responses (e.g., mean intensity).
        
    Returns:
        dict: A dictionary containing the fitted parameters, including IC50.
    """
    if len(concentrations) < 4 or len(responses) < 4:
        return {"error": "Insufficient data for 4PL fit."}
        
    try:
        try:
            from scipy.optimize import curve_fit
            import numpy as np
            
            # Convert to numpy arrays
            x_data = np.array(concentrations)
            y_data = np.array(responses)
            
            p0 = [np.min(y_data), 1, np.median(x_data), np.max(y_data)]
            params, _ = curve_fit(four_param_logistic, x_data, y_data, p0=p0, maxfev=10000)
            
            ic50 = params[2]  # C parameter is the IC50
            
            return {
                'ic50': ic50,
                'min_response': params[0],  # A parameter
                'hill_slope': params[1],    # B parameter
                'max_response': params[3],  # D parameter
                'fit_method': 'scipy_curve_fit'
            }
            
        except ImportError:
            # Fallback to simple estimation if scipy not available
            return estimate_ic50_simple(concentrations, responses)
            
    except RuntimeError:
        return {"error": "Dose-response curve fit failed."}

def estimate_ic50_simple(concentrations, responses):
    """
    Simple IC50 estimation without scipy dependency.
    Finds the concentration closest to 50% response.
    """
    try:
        min_resp = min(responses)
        max_resp = max(responses)
        
        if max_resp == min_resp:
            return {"error": "No response variation in data"}
            
        normalized = [(r - min_resp) / (max_resp - min_resp) * 100 for r in responses]
        
        target = 50.0
        best_idx = 0
        best_diff = abs(normalized[0] - target)
        
        for i, norm_resp in enumerate(normalized):
            diff = abs(norm_resp - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        
        ic50_estimate = concentrations[best_idx]
        
        return {
            'ic50': ic50_estimate,
            'min_response': min_resp,
            'max_response': max_resp,
            'fit_method': 'simple_estimation',
            'closest_response_percent': normalized[best_idx]
        }
        
    except Exception as e:
        return {"error": f"Simple IC50 estimation failed: {str(e)}"}

def array_to_image(array):
    """Convert array back to PIL Image"""
    if not HAS_SCIENTIFIC_LIBS:
        return None

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    # Normalize to 0-255
    min_val, max_val = np.min(array), np.max(array)

    if max_val > min_val:
        normalized = (array - min_val) * 255 / (max_val - min_val)
    else:
        normalized = array

    img = Image.fromarray(normalized.astype(np.uint8))
    return img

def create_label_overlay(image, labels):
    """Create colored overlay of labels on original image"""
    if not HAS_SCIENTIFIC_LIBS:
        return image # return original image if no libs

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Create an RGB image
    overlay = np.stack([image.astype(np.uint8)]*3, axis=-1)

    # Find boundaries and color them red
    boundaries = segmentation.find_boundaries(labels, mode='thick')
    overlay[boundaries] = [255, 0, 0] # Red boundaries

    # Convert to PIL image for sending to frontend
    return Image.fromarray(overlay)
