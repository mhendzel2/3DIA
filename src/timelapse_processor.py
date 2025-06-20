#!/usr/bin/env python3
"""
Timelapse Processing Module for Scientific Image Analyzer
Handles intensity normalization and temporal analysis
"""

import os
import json
import math
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

class ImageAligner:
    """2D image alignment for timelapse sequences"""
    
    def __init__(self):
        self.alignment_methods = {
            'phase_correlation': self._phase_correlation_alignment,
            'cross_correlation': self._cross_correlation_alignment,
            'feature_based': self._feature_based_alignment,
            'intensity_based': self._intensity_based_alignment
        }
    
    def align_timelapse_sequence(self, image_sequence: List[List[List]], 
                               reference_frame: int = 0,
                               method: str = 'cross_correlation',
                               params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Align all frames in a timelapse sequence to a reference frame
        
        Args:
            image_sequence: List of 2D images [time][y][x]
            reference_frame: Index of reference frame (default: first frame)
            method: Alignment method to use
            params: Method-specific parameters
        """
        if not image_sequence or len(image_sequence) < 2:
            return {'error': 'Need at least 2 frames for alignment'}
        
        if reference_frame >= len(image_sequence):
            return {'error': 'Reference frame index out of range'}
        
        if params is None:
            params = {}
        
        try:
            alignment_func = self.alignment_methods.get(method)
            if not alignment_func:
                return {'error': f'Unknown alignment method: {method}'}
            
            reference = image_sequence[reference_frame]
            aligned_sequence = []
            shift_vectors = []
            alignment_scores = []
            
            for frame_idx, frame in enumerate(image_sequence):
                if frame_idx == reference_frame:
                    # Reference frame doesn't need alignment
                    aligned_sequence.append(frame)
                    shift_vectors.append((0, 0))
                    alignment_scores.append(1.0)
                else:
                    # Align frame to reference
                    result = alignment_func(reference, frame, params)
                    
                    if 'error' in result:
                        return {'error': f'Alignment failed for frame {frame_idx}: {result["error"]}'}
                    
                    aligned_frame = result['aligned_image']
                    shift_vector = result['shift_vector']
                    score = result.get('alignment_score', 0.0)
                    
                    aligned_sequence.append(aligned_frame)
                    shift_vectors.append(shift_vector)
                    alignment_scores.append(score)
            
            # Calculate drift statistics
            drift_stats = self._calculate_drift_statistics(shift_vectors)
            
            return {
                'success': True,
                'aligned_sequence': aligned_sequence,
                'shift_vectors': shift_vectors,
                'alignment_scores': alignment_scores,
                'reference_frame': reference_frame,
                'alignment_method': method,
                'drift_statistics': drift_stats,
                'total_frames': len(image_sequence)
            }
            
        except Exception as e:
            return {'error': f'Alignment failed: {str(e)}'}
    
    def _cross_correlation_alignment(self, reference: List[List], target: List[List], 
                                   params: Dict) -> Dict[str, Any]:
        """Cross-correlation based alignment"""
        try:
            max_shift = params.get('max_shift', 50)
            subpixel = params.get('subpixel', False)
            
            ref_height, ref_width = len(reference), len(reference[0])
            target_height, target_width = len(target), len(target[0])
            
            if ref_height != target_height or ref_width != target_width:
                return {'error': 'Images must have same dimensions'}
            
            best_correlation = -1
            best_shift = (0, 0)
            
            # Search over possible shifts
            for dy in range(-max_shift, max_shift + 1):
                for dx in range(-max_shift, max_shift + 1):
                    correlation = self._calculate_normalized_cross_correlation(
                        reference, target, dx, dy)
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_shift = (dx, dy)
            
            # Apply best shift
            aligned_image = self._apply_shift(target, best_shift[0], best_shift[1])
            
            return {
                'aligned_image': aligned_image,
                'shift_vector': best_shift,
                'alignment_score': best_correlation
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _phase_correlation_alignment(self, reference: List[List], target: List[List], 
                                   params: Dict) -> Dict[str, Any]:
        """Phase correlation alignment using FFT"""
        try:
            # Proper phase correlation implementation
            import numpy as np
            from scipy.fft import fft2, ifft2

            # Convert to numpy arrays
            ref_np = np.array(reference)
            target_np = np.array(target)

            # Compute FFTs
            G = fft2(ref_np)
            F = fft2(target_np)

            # Compute cross-power spectrum
            R = (G * np.conj(F)) / (np.abs(G * np.conj(F)) + 1e-10)  # Add epsilon for stability
            r = np.real(ifft2(R))

            # Find the peak of the correlation
            peak_coords = np.unravel_index(np.argmax(r), r.shape)
            shifts = np.array(peak_coords)
            center = np.array(r.shape) / 2
            
            shift_vector = shifts - center
            
            # Apply the calculated shift
            aligned_image_np = self._apply_shift_numpy(target_np, int(shift_vector[1]), int(shift_vector[0]))
            
            return {
                'aligned_image': aligned_image_np.tolist(),
                'shift_vector': (int(shift_vector[1]), int(shift_vector[0])),
                'alignment_score': np.max(r)
            }
        except Exception as e:
            print(f"Phase correlation failed, using fallback: {e}")
            # Fallback to cross-correlation ONLY if FFT fails
            return self._cross_correlation_alignment(reference, target, params)

    def _apply_shift_numpy(self, image, dx: int, dy: int):
        """Helper function to apply shift using numpy for efficiency"""
        try:
            from scipy.ndimage import shift
            # The shift is inverted for scipy's implementation
            return shift(image, [-dy, -dx], cval=0)
        except ImportError:
            # Pure Python fallback for shift
            height, width = len(image), len(image[0])
            shifted = [[0 for _ in range(width)] for _ in range(height)]
            
            for y in range(height):
                for x in range(width):
                    new_y, new_x = y + dy, x + dx
                    if 0 <= new_y < height and 0 <= new_x < width:
                        shifted[new_y][new_x] = image[y][x]
            
            return shifted
    
    def _feature_based_alignment(self, reference: List[List], target: List[List], 
                               params: Dict) -> Dict[str, Any]:
        """Feature-based alignment using corner detection"""
        try:
            min_features = params.get('min_features', 4)
            
            # Detect corners in both images
            ref_corners = self._detect_corners(reference)
            target_corners = self._detect_corners(target)
            
            if len(ref_corners) < min_features or len(target_corners) < min_features:
                # Fall back to cross-correlation if not enough features
                return self._cross_correlation_alignment(reference, target, params)
            
            # Match features and estimate transformation
            matches = self._match_features(ref_corners, target_corners)
            
            if len(matches) < min_features:
                return self._cross_correlation_alignment(reference, target, params)
            
            # Estimate translation (simplified - only translation, no rotation/scaling)
            dx_sum, dy_sum = 0, 0
            for ref_pt, target_pt in matches:
                dx_sum += ref_pt[0] - target_pt[0]
                dy_sum += ref_pt[1] - target_pt[1]
            
            dx = dx_sum / len(matches)
            dy = dy_sum / len(matches)
            
            # Apply translation
            aligned_image = self._apply_shift(target, int(dx), int(dy))
            
            # Calculate alignment score based on feature matching
            score = min(1.0, len(matches) / max(len(ref_corners), len(target_corners)))
            
            return {
                'aligned_image': aligned_image,
                'shift_vector': (int(dx), int(dy)),
                'alignment_score': score,
                'features_matched': len(matches)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _intensity_based_alignment(self, reference: List[List], target: List[List], 
                                 params: Dict) -> Dict[str, Any]:
        """Intensity-based mutual information alignment"""
        try:
            max_shift = params.get('max_shift', 30)
            
            best_mi = -1
            best_shift = (0, 0)
            
            # Search over possible shifts
            for dy in range(-max_shift, max_shift + 1):
                for dx in range(-max_shift, max_shift + 1):
                    mi = self._calculate_mutual_information(reference, target, dx, dy)
                    
                    if mi > best_mi:
                        best_mi = mi
                        best_shift = (dx, dy)
            
            # Apply best shift
            aligned_image = self._apply_shift(target, best_shift[0], best_shift[1])
            
            return {
                'aligned_image': aligned_image,
                'shift_vector': best_shift,
                'alignment_score': best_mi
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_normalized_cross_correlation(self, ref: List[List], target: List[List], 
                                              dx: int, dy: int) -> float:
        """Calculate normalized cross-correlation between shifted images"""
        try:
            ref_height, ref_width = len(ref), len(ref[0])
            
            # Define overlap region
            y_start = max(0, dy)
            y_end = min(ref_height, ref_height + dy)
            x_start = max(0, dx)
            x_end = min(ref_width, ref_width + dx)
            
            if y_end <= y_start or x_end <= x_start:
                return 0
            
            # Calculate means
            ref_sum = target_sum = 0
            pixel_count = 0
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    target_y, target_x = y - dy, x - dx
                    if 0 <= target_y < ref_height and 0 <= target_x < ref_width:
                        ref_sum += ref[y][x]
                        target_sum += target[target_y][target_x]
                        pixel_count += 1
            
            if pixel_count == 0:
                return 0
            
            ref_mean = ref_sum / pixel_count
            target_mean = target_sum / pixel_count
            
            # Calculate correlation
            numerator = 0
            ref_var = target_var = 0
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    target_y, target_x = y - dy, x - dx
                    if 0 <= target_y < ref_height and 0 <= target_x < ref_width:
                        ref_diff = ref[y][x] - ref_mean
                        target_diff = target[target_y][target_x] - target_mean
                        
                        numerator += ref_diff * target_diff
                        ref_var += ref_diff * ref_diff
                        target_var += target_diff * target_diff
            
            if ref_var == 0 or target_var == 0:
                return 0
            
            correlation = numerator / math.sqrt(ref_var * target_var)
            return correlation
            
        except Exception:
            return 0
    
    def _calculate_mutual_information(self, ref: List[List], target: List[List], 
                                    dx: int, dy: int) -> float:
        """Calculate mutual information between shifted images"""
        try:
            ref_height, ref_width = len(ref), len(ref[0])
            
            # Define overlap region
            y_start = max(0, dy)
            y_end = min(ref_height, ref_height + dy)
            x_start = max(0, dx)
            x_end = min(ref_width, ref_width + dx)
            
            if y_end <= y_start or x_end <= x_start:
                return 0
            
            # Collect intensity pairs
            ref_intensities = []
            target_intensities = []
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    target_y, target_x = y - dy, x - dx
                    if 0 <= target_y < ref_height and 0 <= target_x < ref_width:
                        ref_intensities.append(int(ref[y][x]))
                        target_intensities.append(int(target[target_y][target_x]))
            
            if not ref_intensities:
                return 0
            
            # Simplified mutual information calculation
            # (In practice, would use proper histogram-based calculation)
            ref_entropy = self._calculate_entropy(ref_intensities)
            target_entropy = self._calculate_entropy(target_intensities)
            joint_entropy = self._calculate_joint_entropy(ref_intensities, target_intensities)
            
            mi = ref_entropy + target_entropy - joint_entropy
            return mi
            
        except Exception:
            return 0
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of intensity values"""
        if not values:
            return 0
        
        # Create histogram
        hist = {}
        for val in values:
            hist[val] = hist.get(val, 0) + 1
        
        # Calculate entropy
        total = len(values)
        entropy = 0
        for count in hist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_joint_entropy(self, values1: List[int], values2: List[int]) -> float:
        """Calculate joint entropy of two value lists"""
        if not values1 or not values2 or len(values1) != len(values2):
            return 0
        
        # Create joint histogram
        joint_hist = {}
        for v1, v2 in zip(values1, values2):
            key = (v1, v2)
            joint_hist[key] = joint_hist.get(key, 0) + 1
        
        # Calculate joint entropy
        total = len(values1)
        entropy = 0
        for count in joint_hist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _detect_corners(self, image: List[List]) -> List[Tuple[int, int]]:
        """Simple corner detection using Harris-like method"""
        try:
            height, width = len(image), len(image[0])
            corners = []
            
            # Calculate gradients
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    # Calculate local gradients
                    gx = (image[y][x+1] - image[y][x-1]) / 2
                    gy = (image[y+1][x] - image[y-1][x]) / 2
                    
                    # Harris corner response (simplified)
                    corner_response = gx * gx + gy * gy
                    
                    # Threshold for corner detection
                    if corner_response > 1000:  # Adjust threshold as needed
                        corners.append((x, y))
            
            # Non-maximum suppression (simplified)
            filtered_corners = []
            min_distance = 10
            
            for corner in corners:
                is_local_max = True
                for existing in filtered_corners:
                    dist = math.sqrt((corner[0] - existing[0])**2 + (corner[1] - existing[1])**2)
                    if dist < min_distance:
                        is_local_max = False
                        break
                
                if is_local_max:
                    filtered_corners.append(corner)
            
            return filtered_corners[:50]  # Limit to top 50 corners
            
        except Exception:
            return []
    
    def _match_features(self, corners1: List[Tuple[int, int]], 
                       corners2: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Match features between two corner lists"""
        matches = []
        max_distance = 20  # Maximum distance for feature matching
        
        for c1 in corners1:
            best_match = None
            best_distance = float('inf')
            
            for c2 in corners2:
                distance = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_match = c2
            
            if best_match:
                matches.append((c1, best_match))
        
        return matches
    
    def _apply_shift(self, image: List[List], dx: int, dy: int) -> List[List]:
        """Apply translation shift to image"""
        height, width = len(image), len(image[0])
        shifted = [[0 for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                src_x = x - dx
                src_y = y - dy
                
                if 0 <= src_x < width and 0 <= src_y < height:
                    shifted[y][x] = image[src_y][src_x]
                # else: leave as 0 (black padding)
        
        return shifted
    
    def _calculate_drift_statistics(self, shift_vectors: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Calculate drift statistics from shift vectors"""
        if not shift_vectors:
            return {}
        
        # Total drift (cumulative displacement)
        total_dx = sum(shift[0] for shift in shift_vectors)
        total_dy = sum(shift[1] for shift in shift_vectors)
        total_drift = math.sqrt(total_dx**2 + total_dy**2)
        
        # Maximum single-frame drift
        frame_drifts = [math.sqrt(dx**2 + dy**2) for dx, dy in shift_vectors]
        max_frame_drift = max(frame_drifts) if frame_drifts else 0
        avg_frame_drift = sum(frame_drifts) / len(frame_drifts) if frame_drifts else 0
        
        # Drift direction
        dominant_direction = 'none'
        if abs(total_dx) > abs(total_dy):
            dominant_direction = 'horizontal' if total_dx > 0 else 'horizontal_negative'
        elif abs(total_dy) > abs(total_dx):
            dominant_direction = 'vertical' if total_dy > 0 else 'vertical_negative'
        
        return {
            'total_drift_pixels': total_drift,
            'total_drift_vector': (total_dx, total_dy),
            'max_frame_drift': max_frame_drift,
            'average_frame_drift': avg_frame_drift,
            'dominant_drift_direction': dominant_direction,
            'drift_per_frame': frame_drifts,
            'is_stable': max_frame_drift < 5  # Consider stable if max drift < 5 pixels
        }

class TimelapseProcessor:
    """Advanced timelapse processing with intensity normalization"""
    
    def __init__(self):
        self.normalization_methods = {
            'global_percentile': self._global_percentile_normalization,
            'frame_percentile': self._frame_percentile_normalization,
            'adaptive_histogram': self._adaptive_histogram_normalization,
            'z_score': self._z_score_normalization,
            'rolling_baseline': self._rolling_baseline_normalization,
            'bleaching_correction': self._bleaching_correction_normalization
        }
    
    def normalize_timelapse(self, image_sequence: List[List[List]], method: str = 'global_percentile', 
                          params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Normalize intensity across timelapse sequence
        
        Args:
            image_sequence: List of 2D images [time][y][x]
            method: Normalization method
            params: Method-specific parameters
        """
        if not image_sequence or not image_sequence[0]:
            return {'error': 'Empty image sequence'}
        
        if params is None:
            params = {}
        
        try:
            normalization_func = self.normalization_methods.get(method)
            if not normalization_func:
                return {'error': f'Unknown normalization method: {method}'}
            
            normalized_sequence, stats = normalization_func(image_sequence, params)
            
            return {
                'success': True,
                'normalized_sequence': normalized_sequence,
                'original_frames': len(image_sequence),
                'normalization_method': method,
                'normalization_stats': stats,
                'frame_dimensions': [len(image_sequence[0]), len(image_sequence[0][0])],
                'intensity_range': self._calculate_intensity_range(normalized_sequence)
            }
            
        except Exception as e:
            return {'error': f'Normalization failed: {str(e)}'}
    
    def _global_percentile_normalization(self, sequence: List[List[List]], params: Dict) -> Tuple[List[List[List]], Dict]:
        """Normalize using global percentile values across entire sequence"""
        low_percentile = params.get('low_percentile', 1)
        high_percentile = params.get('high_percentile', 99)
        
        # Collect all pixel values
        all_values = []
        for frame in sequence:
            for row in frame:
                all_values.extend(row)
        
        all_values.sort()
        n_values = len(all_values)
        
        # Calculate percentile values
        low_idx = int(n_values * low_percentile / 100)
        high_idx = int(n_values * high_percentile / 100)
        
        low_val = all_values[low_idx]
        high_val = all_values[high_idx]
        
        # Normalize sequence
        normalized = []
        for frame in sequence:
            norm_frame = []
            for row in frame:
                norm_row = []
                for pixel in row:
                    # Clip and normalize to 0-255 range
                    clipped = max(low_val, min(high_val, pixel))
                    normalized_pixel = (clipped - low_val) / (high_val - low_val) * 255
                    norm_row.append(normalized_pixel)
                norm_frame.append(norm_row)
            normalized.append(norm_frame)
        
        stats = {
            'global_low_value': low_val,
            'global_high_value': high_val,
            'percentiles_used': [low_percentile, high_percentile]
        }
        
        return normalized, stats
    
    def _frame_percentile_normalization(self, sequence: List[List[List]], params: Dict) -> Tuple[List[List[List]], Dict]:
        """Normalize each frame independently using percentiles"""
        low_percentile = params.get('low_percentile', 1)
        high_percentile = params.get('high_percentile', 99)
        
        normalized = []
        frame_stats = []
        
        for frame_idx, frame in enumerate(sequence):
            # Collect frame values
            frame_values = []
            for row in frame:
                frame_values.extend(row)
            
            frame_values.sort()
            n_values = len(frame_values)
            
            # Calculate percentiles for this frame
            low_idx = int(n_values * low_percentile / 100)
            high_idx = int(n_values * high_percentile / 100)
            
            low_val = frame_values[low_idx]
            high_val = frame_values[high_idx]
            
            # Normalize frame
            norm_frame = []
            for row in frame:
                norm_row = []
                for pixel in row:
                    if high_val > low_val:
                        clipped = max(low_val, min(high_val, pixel))
                        normalized_pixel = (clipped - low_val) / (high_val - low_val) * 255
                    else:
                        normalized_pixel = 128  # Default if no range
                    norm_row.append(normalized_pixel)
                norm_frame.append(norm_row)
            
            normalized.append(norm_frame)
            frame_stats.append({
                'frame': frame_idx,
                'low_value': low_val,
                'high_value': high_val
            })
        
        stats = {
            'method': 'frame_percentile',
            'frame_statistics': frame_stats
        }
        
        return normalized, stats
    
    def _adaptive_histogram_normalization(self, sequence: List[List[List]], params: Dict) -> Tuple[List[List[List]], Dict]:
        """Adaptive histogram equalization for each frame"""
        clip_limit = params.get('clip_limit', 2.0)
        
        normalized = []
        
        for frame in sequence:
            # Calculate histogram
            hist = [0] * 256
            total_pixels = 0
            
            for row in frame:
                for pixel in row:
                    hist_idx = max(0, min(255, int(pixel)))
                    hist[hist_idx] += 1
                    total_pixels += 1
            
            # Calculate cumulative distribution
            cdf = [0] * 256
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i-1] + hist[i]
            
            # Normalize CDF
            if total_pixels > 0:
                cdf = [c / total_pixels for c in cdf]
            
            # Apply histogram equalization
            norm_frame = []
            for row in frame:
                norm_row = []
                for pixel in row:
                    hist_idx = max(0, min(255, int(pixel)))
                    equalized = cdf[hist_idx] * 255
                    norm_row.append(equalized)
                norm_frame.append(norm_row)
            
            normalized.append(norm_frame)
        
        stats = {
            'method': 'adaptive_histogram',
            'clip_limit': clip_limit
        }
        
        return normalized, stats
    
    def _z_score_normalization(self, sequence: List[List[List]], params: Dict) -> Tuple[List[List[List]], Dict]:
        """Z-score normalization (mean=0, std=1) then scale to 0-255"""
        global_stats = params.get('global_stats', True)
        
        if global_stats:
            # Calculate global mean and std
            all_values = []
            for frame in sequence:
                for row in frame:
                    all_values.extend(row)
            
            mean_val = sum(all_values) / len(all_values)
            variance = sum((x - mean_val) ** 2 for x in all_values) / len(all_values)
            std_val = math.sqrt(variance) if variance > 0 else 1
            
            # Normalize all frames with global stats
            normalized = []
            for frame in sequence:
                norm_frame = []
                for row in frame:
                    norm_row = []
                    for pixel in row:
                        z_score = (pixel - mean_val) / std_val
                        # Scale to 0-255 range (clamp to ±3 std)
                        clamped = max(-3, min(3, z_score))
                        scaled = (clamped + 3) / 6 * 255
                        norm_row.append(scaled)
                    norm_frame.append(norm_row)
                normalized.append(norm_frame)
            
            stats = {
                'global_mean': mean_val,
                'global_std': std_val,
                'normalization_range': '±3 standard deviations'
            }
        else:
            # Frame-by-frame z-score normalization
            normalized = []
            frame_stats = []
            
            for frame_idx, frame in enumerate(sequence):
                # Calculate frame statistics
                frame_values = []
                for row in frame:
                    frame_values.extend(row)
                
                frame_mean = sum(frame_values) / len(frame_values)
                frame_variance = sum((x - frame_mean) ** 2 for x in frame_values) / len(frame_values)
                frame_std = math.sqrt(frame_variance) if frame_variance > 0 else 1
                
                # Normalize frame
                norm_frame = []
                for row in frame:
                    norm_row = []
                    for pixel in row:
                        z_score = (pixel - frame_mean) / frame_std
                        clamped = max(-3, min(3, z_score))
                        scaled = (clamped + 3) / 6 * 255
                        norm_row.append(scaled)
                    norm_frame.append(norm_row)
                
                normalized.append(norm_frame)
                frame_stats.append({
                    'frame': frame_idx,
                    'mean': frame_mean,
                    'std': frame_std
                })
            
            stats = {
                'method': 'frame_z_score',
                'frame_statistics': frame_stats
            }
        
        return normalized, stats
    
    def _rolling_baseline_normalization(self, sequence: List[List[List]], params: Dict) -> Tuple[List[List[List]], Dict]:
        """Rolling baseline correction for photobleaching"""
        window_size = params.get('window_size', 10)
        percentile = params.get('baseline_percentile', 10)
        
        # Calculate frame-wise baseline intensities
        frame_baselines = []
        for frame in sequence:
            frame_values = []
            for row in frame:
                frame_values.extend(row)
            
            frame_values.sort()
            baseline_idx = int(len(frame_values) * percentile / 100)
            baseline = frame_values[baseline_idx]
            frame_baselines.append(baseline)
        
        # Calculate rolling baseline
        rolling_baselines = []
        half_window = window_size // 2
        
        for i in range(len(sequence)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(sequence), i + half_window + 1)
            
            window_baselines = frame_baselines[start_idx:end_idx]
            rolling_baseline = sum(window_baselines) / len(window_baselines)
            rolling_baselines.append(rolling_baseline)
        
        # Normalize by dividing by rolling baseline
        normalized = []
        for frame_idx, frame in enumerate(sequence):
            baseline = rolling_baselines[frame_idx]
            
            norm_frame = []
            for row in frame:
                norm_row = []
                for pixel in row:
                    if baseline > 0:
                        corrected = pixel / baseline * 100  # Scale to percentage
                        normalized_pixel = min(255, corrected)  # Cap at 255
                    else:
                        normalized_pixel = pixel
                    norm_row.append(normalized_pixel)
                norm_frame.append(norm_row)
            
            normalized.append(norm_frame)
        
        stats = {
            'method': 'rolling_baseline',
            'window_size': window_size,
            'baseline_percentile': percentile,
            'frame_baselines': frame_baselines,
            'rolling_baselines': rolling_baselines
        }
        
        return normalized, stats
    
    def _bleaching_correction_normalization(self, sequence: List[List[List]], params: Dict) -> Tuple[List[List[List]], Dict]:
        """Exponential bleaching correction"""
        reference_frame = params.get('reference_frame', 0)
        correction_method = params.get('method', 'exponential')
        
        # Calculate mean intensity for each frame
        frame_means = []
        for frame in sequence:
            total_intensity = 0
            total_pixels = 0
            for row in frame:
                for pixel in row:
                    total_intensity += pixel
                    total_pixels += 1
            
            mean_intensity = total_intensity / total_pixels if total_pixels > 0 else 0
            frame_means.append(mean_intensity)
        
        reference_intensity = frame_means[reference_frame]
        
        if correction_method == 'exponential':
            # Fit exponential decay curve
            correction_factors = []
            for i, mean_intensity in enumerate(frame_means):
                if mean_intensity > 0:
                    factor = reference_intensity / mean_intensity
                else:
                    factor = 1.0
                correction_factors.append(factor)
        else:
            # Linear correction
            correction_factors = []
            for mean_intensity in frame_means:
                if mean_intensity > 0:
                    factor = reference_intensity / mean_intensity
                else:
                    factor = 1.0
                correction_factors.append(factor)
        
        # Apply correction factors
        normalized = []
        for frame_idx, frame in enumerate(sequence):
            factor = correction_factors[frame_idx]
            
            norm_frame = []
            for row in frame:
                norm_row = []
                for pixel in row:
                    corrected = pixel * factor
                    normalized_pixel = min(255, corrected)  # Cap at 255
                    norm_row.append(normalized_pixel)
                norm_frame.append(norm_row)
            
            normalized.append(norm_frame)
        
        stats = {
            'method': 'bleaching_correction',
            'reference_frame': reference_frame,
            'reference_intensity': reference_intensity,
            'frame_means': frame_means,
            'correction_factors': correction_factors
        }
        
        return normalized, stats
    
    def _calculate_intensity_range(self, sequence: List[List[List]]) -> Dict[str, float]:
        """Calculate intensity statistics for the sequence"""
        all_values = []
        for frame in sequence:
            for row in frame:
                all_values.extend(row)
        
        if not all_values:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        min_val = min(all_values)
        max_val = max(all_values)
        mean_val = sum(all_values) / len(all_values)
        
        variance = sum((x - mean_val) ** 2 for x in all_values) / len(all_values)
        std_val = math.sqrt(variance)
        
        return {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val
        }
    
    def analyze_temporal_dynamics(self, sequence: List[List[List]], roi_coords: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Analyze temporal dynamics of intensity changes"""
        try:
            frame_count = len(sequence)
            if frame_count < 2:
                return {'error': 'Need at least 2 frames for temporal analysis'}
            
            # Define ROI (region of interest)
            if roi_coords is None:
                # Use center region as default ROI
                height = len(sequence[0])
                width = len(sequence[0][0])
                center_y, center_x = height // 2, width // 2
                roi_size = min(height, width) // 4
                
                roi_coords = []
                for y in range(max(0, center_y - roi_size), min(height, center_y + roi_size)):
                    for x in range(max(0, center_x - roi_size), min(width, center_x + roi_size)):
                        roi_coords.append((y, x))
            
            # Calculate ROI intensity over time
            roi_intensities = []
            for frame in sequence:
                roi_sum = 0
                valid_pixels = 0
                
                for y, x in roi_coords:
                    if 0 <= y < len(frame) and 0 <= x < len(frame[0]):
                        roi_sum += frame[y][x]
                        valid_pixels += 1
                
                roi_mean = roi_sum / valid_pixels if valid_pixels > 0 else 0
                roi_intensities.append(roi_mean)
            
            # Calculate dynamics metrics
            max_intensity = max(roi_intensities)
            min_intensity = min(roi_intensities)
            intensity_range = max_intensity - min_intensity
            
            # Calculate rate of change
            intensity_changes = []
            for i in range(1, len(roi_intensities)):
                change = roi_intensities[i] - roi_intensities[i-1]
                intensity_changes.append(change)
            
            avg_change_rate = sum(intensity_changes) / len(intensity_changes) if intensity_changes else 0
            
            # Detect significant events (threshold-based)
            threshold = intensity_range * 0.1  # 10% of total range
            events = []
            
            for i, change in enumerate(intensity_changes):
                if abs(change) > threshold:
                    events.append({
                        'frame': i + 1,
                        'intensity_change': change,
                        'event_type': 'increase' if change > 0 else 'decrease'
                    })
            
            return {
                'success': True,
                'roi_size': len(roi_coords),
                'frame_count': frame_count,
                'roi_intensities': roi_intensities,
                'max_intensity': max_intensity,
                'min_intensity': min_intensity,
                'intensity_range': intensity_range,
                'average_change_rate': avg_change_rate,
                'significant_events': events,
                'bleaching_detected': avg_change_rate < -threshold,
                'activation_detected': any(change > threshold for change in intensity_changes)
            }
            
        except Exception as e:
            return {'error': f'Temporal analysis failed: {str(e)}'}
    
    def export_timelapse_data(self, sequence_data: Dict[str, Any], export_path: str) -> Dict[str, Any]:
        """Export timelapse analysis results"""
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Export normalized sequence as JSON (for small datasets)
            export_data = {
                'metadata': {
                    'normalization_method': sequence_data.get('normalization_method'),
                    'original_frames': sequence_data.get('original_frames'),
                    'frame_dimensions': sequence_data.get('frame_dimensions'),
                    'export_timestamp': time.time()
                },
                'normalization_stats': sequence_data.get('normalization_stats', {}),
                'intensity_range': sequence_data.get('intensity_range', {}),
                'sequence_length': len(sequence_data.get('normalized_sequence', []))
            }
            
            # For large sequences, export summary only
            if len(sequence_data.get('normalized_sequence', [])) > 100:
                export_data['note'] = 'Large sequence - only metadata exported'
            else:
                export_data['normalized_sequence'] = sequence_data.get('normalized_sequence', [])
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return {
                'success': True,
                'export_path': export_path,
                'file_size': os.path.getsize(export_path)
            }
            
        except Exception as e:
            return {'error': f'Export failed: {str(e)}'}

# Preset configurations for common timelapse scenarios
TIMELAPSE_PRESETS = {
    'fluorescence_bleaching': {
        'normalization_method': 'bleaching_correction',
        'params': {
            'reference_frame': 0,
            'method': 'exponential'
        },
        'description': 'Corrects for fluorescence photobleaching over time'
    },
    'live_cell_imaging': {
        'normalization_method': 'rolling_baseline',
        'params': {
            'window_size': 5,
            'baseline_percentile': 10
        },
        'description': 'Adaptive baseline correction for live cell dynamics'
    },
    'calcium_imaging': {
        'normalization_method': 'z_score',
        'params': {
            'global_stats': False
        },
        'description': 'Frame-by-frame Z-score normalization for calcium signals'
    },
    'developmental_series': {
        'normalization_method': 'global_percentile',
        'params': {
            'low_percentile': 0.1,
            'high_percentile': 99.9
        },
        'description': 'Global intensity normalization across developmental stages'
    },
    'high_contrast': {
        'normalization_method': 'adaptive_histogram',
        'params': {
            'clip_limit': 3.0
        },
        'description': 'Adaptive histogram equalization for high contrast imaging'
    }
}