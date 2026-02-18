"""
Scientific Image Analysis Application
Comprehensive microscopy analysis suite replicating Napari Hub plugin functionality
"""

import os
import io
import base64
import json
import time
from pathlib import Path
from collections import OrderedDict
from threading import Lock

# Flask with fallback
try:
    from flask import Flask, render_template, request, jsonify
    from werkzeug.utils import secure_filename
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Flask not available - creating standalone analyzer")

# PIL for image handling
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not available")

try:
    from batch_processor import BatchProcessor, WORKFLOW_TEMPLATES
    HAS_BATCH_PROCESSOR = True
except ImportError:
    HAS_BATCH_PROCESSOR = False
    print("BatchProcessor not available - batch processing disabled")

from utils import analysis_utils as au
import numpy as np
from scipy import ndimage as ndi


class LRUCache:
    """
    Thread-safe LRU cache for session data with memory limits.
    Automatically evicts oldest entries when limits are exceeded.
    """
    def __init__(self, max_entries=50, max_memory_mb=500):
        """
        Initialize LRU cache.
        
        Args:
            max_entries: Maximum number of cache entries
            max_memory_mb: Maximum estimated memory usage in MB
        """
        self._cache = OrderedDict()
        self._lock = Lock()
        self._max_entries = max_entries
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._access_times = {}
    
    def _estimate_size(self, value):
        """Estimate memory size of a cache value in bytes"""
        size = 0
        if isinstance(value, dict):
            for k, v in value.items():
                size += self._estimate_size(v)
        elif isinstance(value, np.ndarray):
            size += value.nbytes
        elif isinstance(value, (list, tuple)):
            for item in value:
                size += self._estimate_size(item)
        elif isinstance(value, str):
            size += len(value)
        else:
            size += 100  # Default estimate for other types
        return size
    
    def _get_total_memory(self):
        """Get total estimated memory usage"""
        total = 0
        for value in self._cache.values():
            total += self._estimate_size(value)
        return total
    
    def _evict_if_needed(self):
        """Evict oldest entries if limits exceeded"""
        # Evict by entry count
        while len(self._cache) > self._max_entries:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            if oldest_key in self._access_times:
                del self._access_times[oldest_key]
            print(f"Cache evicted session (max entries): {oldest_key}")
        
        # Evict by memory (check periodically)
        while self._get_total_memory() > self._max_memory_bytes and len(self._cache) > 0:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            if oldest_key in self._access_times:
                del self._access_times[oldest_key]
            print(f"Cache evicted session (memory limit): {oldest_key}")
    
    def get(self, key, default=None):
        """Get value and update access time"""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._access_times[key] = time.time()
                return self._cache[key]
            return default
    
    def set(self, key, value):
        """Set value and evict if needed"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._evict_if_needed()
    
    def delete(self, key):
        """Delete a specific entry"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
    
    def __contains__(self, key):
        with self._lock:
            return key in self._cache
    
    def __getitem__(self, key):
        result = self.get(key)
        if result is None and key not in self._cache:
            raise KeyError(key)
        return result
    
    def __setitem__(self, key, value):
        self.set(key, value)
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self):
        """Get cache statistics"""
        with self._lock:
            return {
                'entries': len(self._cache),
                'max_entries': self._max_entries,
                'estimated_memory_mb': self._get_total_memory() / (1024 * 1024),
                'max_memory_mb': self._max_memory_bytes / (1024 * 1024)
            }


# Web application (if Flask is available)
if HAS_FLASK:
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
    
    # Create uploads directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Global storage with LRU cache for memory management
    analysis_cache = LRUCache(max_entries=50, max_memory_mb=500)
    
    if HAS_BATCH_PROCESSOR:
        batch_processor = BatchProcessor()
    else:
        batch_processor = None
    
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
            
            # Load image using the new utility
            image_array = au.load_image(str(file_path))
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
            img_pil = au.array_to_image(image_array)
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
                'shape': image_array.shape,
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
            session_data = analysis_cache.get(session_id)
            if session_data is None:
                return jsonify({'error': 'Session data not found'}), 400
            
            image = session_data['original_image']
            
            if method == 'cellpose':
                labels = au.segment_cellpose(image)
            elif method == 'stardist':
                labels = au.segment_stardist(image)
            elif method == 'watershed':
                labels = au.segment_watershed(image)
            else:
                return jsonify({'error': 'Unknown method'}), 400

            if labels is None:
                return jsonify({'error': f'{method} segmentation failed. Check server logs.'}), 500
            
            # Store results
            session_data['processed_images'][f'labels_{method}'] = labels
            
            # Generate measurements (returns a dict, not DataFrame)
            measurements_dict = au.calculate_object_statistics(labels, image)
            
            # Convert dict of lists to list of dicts for JSON serialization
            if 'label' in measurements_dict:
                num_objects = len(measurements_dict.get('label', []))
                measurements = []
                for i in range(num_objects):
                    obj_data = {}
                    for key, values in measurements_dict.items():
                        if isinstance(values, list) and len(values) > i:
                            obj_data[key] = values[i]
                    measurements.append(obj_data)
            else:
                measurements = [measurements_dict]
            
            session_data['results'][f'measurements_{method}'] = measurements
            analysis_cache.set(session_id, session_data)  # Update cache
            
            # Create visualization
            overlay_pil = au.create_label_overlay(image, labels)
            
            if overlay_pil:
                overlay_pil.thumbnail((400, 400))
                buffer = io.BytesIO()
                overlay_pil.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                overlay_data = f"data:image/png;base64,{img_str}"
            else:
                overlay_data = ""
            
            return jsonify({
                'success': True,
                'segmentation_data': overlay_data,
                'object_count': len(measurements),
                'measurements': measurements[:10]
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
            session_data = analysis_cache.get(session_id)
            if session_data is None:
                return jsonify({'error': 'Session data not found'}), 400
            
            image = session_data['original_image']
            
            if analysis_type == 'intensity':
                results = au.calculate_intensity_statistics(image)
            elif analysis_type == 'colocalization':
                # Create synthetic second channel for demo
                channel2 = ndi.gaussian_filter(image, sigma=2)
                results = au.calculate_colocalization_coefficients(image, channel2)
            else:
                results = {'error': 'Unknown analysis type'}
            
            session_data['results'][f'analysis_{analysis_type}'] = results
            analysis_cache.set(session_id, session_data)  # Update cache
            
            return jsonify({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    @app.route('/track', methods=['POST'])
    def track_objects():
        """Object tracking endpoint using the core Hungarian tracking backend."""
        payload = request.get_json() or {}
        session_id = payload.get('session_id')
        labels_key = payload.get('labels_key')
        max_distance = float(payload.get('max_distance', 50.0))

        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        if session_id not in analysis_cache:
            return jsonify({'error': 'Invalid session'}), 400

        session_data = analysis_cache.get(session_id)
        if session_data is None:
            return jsonify({'error': 'Session data not found'}), 400

        try:
            labels_sequence = None

            if labels_key:
                candidate = session_data.get('processed_images', {}).get(str(labels_key))
                labels_sequence = _coerce_labels_sequence(candidate)
            else:
                processed = session_data.get('processed_images', {})
                for key, value in processed.items():
                    if str(key).startswith('labels_'):
                        labels_sequence = _coerce_labels_sequence(value)
                        if labels_sequence:
                            break

            if not labels_sequence:
                image = np.asarray(session_data.get('original_image'))
                labels_sequence = _labels_from_timeseries(image)

            if not labels_sequence or len(labels_sequence) < 2:
                return jsonify({
                    'error': (
                        'Tracking requires at least two timepoints of labels. '
                        'Provide a time-series labels array or a time-resolved image.'
                    )
                }), 400

            from pymaris.backends import DEFAULT_REGISTRY  # local import keeps Flask startup lightweight

            backend = DEFAULT_REGISTRY.get_tracking('hungarian')
            result = backend.track(labels_sequence, max_distance=max_distance)

            track_payload = {
                'table': result.table,
                'metadata': result.metadata,
                'track_count': int(result.table.get('total_tracks', 0)),
                'track_lengths': result.table.get('track_lengths', []),
                'napari_tracks_preview': (
                    result.tracks.get('napari_tracks', np.empty((0, 0)))[:200].tolist()
                    if isinstance(result.tracks, dict)
                    else []
                ),
            }
            session_data.setdefault('results', {})['tracking'] = track_payload
            analysis_cache.set(session_id, session_data)

            return jsonify({
                'success': True,
                'tracking': track_payload,
            })
        except Exception as exc:
            return jsonify({'error': f'Tracking failed: {exc}'}), 500


    def _coerce_labels_sequence(value):
        """Return a list of label images from list/tuple/ndarray inputs."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            output = [np.asarray(item) for item in value]
            return output if len(output) >= 2 else None
        array = np.asarray(value)
        if array.ndim < 3:
            return None
        return [np.asarray(array[index]) for index in range(array.shape[0])]


    def _labels_from_timeseries(image):
        """Generate coarse labels from time-series intensity data as tracking fallback."""
        if image.ndim < 3:
            return None
        frames = [np.asarray(image[index], dtype=float) for index in range(image.shape[0])]
        labels_sequence = []
        for frame in frames:
            threshold = float(np.percentile(frame, 90))
            mask = frame > threshold
            labels, _ = ndi.label(mask)
            labels_sequence.append(np.asarray(labels, dtype=np.int32))
        return labels_sequence

    @app.route('/api/cache/stats', methods=['GET'])
    def get_cache_stats():
        """Get cache statistics for monitoring"""
        try:
            stats = analysis_cache.get_stats()
            return jsonify({
                'success': True,
                'cache_stats': stats
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/cache/clear', methods=['POST'])
    def clear_cache():
        """Clear all cached session data"""
        try:
            analysis_cache.clear()
            return jsonify({
                'success': True,
                'message': 'Cache cleared successfully'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/create_test', methods=['POST'])
    def create_test_image_route(): # Renamed to avoid conflict
        """Create a test image for demonstration"""
        try:
            # Create test image
            image = np.zeros((400,400), dtype=np.uint8)
            image[100:200, 100:200] = 128
            image[250:350, 250:350] = 255
            
            # Convert to PIL and save
            img_pil = au.array_to_image(image)
            if img_pil:
                img_pil.thumbnail((400, 400))
                buffer = io.BytesIO()
                img_pil.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                image_data = f"data:image/png;base64,{img_str}"
                
                # Create session
                session_id = "test_image_session"
                analysis_cache[session_id] = {
                    'original_image': image,
                    'processed_images': {},
                    'results': {}
                }
                
                return jsonify({
                    'success': True,
                    'image_data': image_data,
                    'info': {
                        'filename': 'test_image.png',
                        'shape': image.shape,
                        'session_id': session_id
                    }
                })
            else:
                return jsonify({'error': 'Could not create test image'}), 500
                
        except Exception as e:
            return jsonify({'error': f'Test image creation failed: {str(e)}'}), 500
    
    @app.route('/api/batch/process', methods=['POST'])
    def process_batch_files():
        """Handle batch processing of multiple files."""
        if not HAS_BATCH_PROCESSOR:
            return jsonify({'error': 'Batch processing not available'}), 503
            
        data = request.get_json()
        files = data.get('files')
        workflow_name = data.get('workflow', 'cell_counting')

        if not files:
            return jsonify({'error': 'No files provided for batch processing'}), 400

        workflow_config = WORKFLOW_TEMPLATES.get(workflow_name)
        if not workflow_config:
            return jsonify({'error': f'Workflow "{workflow_name}" not found'}), 400

        try:
            batch_id = batch_processor.create_batch_session(files, workflow_config)
            batch_processor.process_batch(batch_id)

            return jsonify({
                'success': True,
                'message': 'Batch processing started.',
                'batch_id': batch_id,
                'workflow': workflow_name,
                'file_count': len(files)
            })
        except Exception as e:
            return jsonify({'error': f'Failed to start batch process: {str(e)}'}), 500

    @app.route('/api/batch/status/<batch_id>', methods=['GET'])
    def get_batch_status(batch_id):
        """Get the status of a running batch process."""
        if not HAS_BATCH_PROCESSOR:
            return jsonify({'error': 'Batch processing not available'}), 503
            
        status = batch_processor.get_batch_status(batch_id)
        if 'error' in status:
            return jsonify(status), 404
        return jsonify(status)

    @app.route('/api/batch/results/<batch_id>', methods=['GET'])
    def get_batch_results(batch_id):
        """Get the results of a completed batch process."""
        if not HAS_BATCH_PROCESSOR:
            return jsonify({'error': 'Batch processing not available'}), 503
            
        results = batch_processor.get_batch_results(batch_id)
        if 'error' in results:
            return jsonify(results), 404
        return jsonify(results)
    
    @app.route('/api/workflows', methods=['GET'])
    def get_available_workflows():
        """Get list of available batch processing workflows."""
        if not HAS_BATCH_PROCESSOR:
            return jsonify({'error': 'Batch processing not available'}), 503
            
        workflows = {}
        for name, config in WORKFLOW_TEMPLATES.items():
            workflows[name] = {
                'name': config.get('name', name),
                'description': config.get('description', 'No description available')
            }
        return jsonify({'workflows': workflows})

def main():
    """Main application entry point"""
    if HAS_FLASK:
        print("Starting Scientific Image Analyzer web application...")
        print("Using robust, library-based analysis functions.")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Flask not available. Cannot run web application.")

if __name__ == '__main__':
    main()
