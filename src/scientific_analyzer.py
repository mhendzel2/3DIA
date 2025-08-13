"""
Scientific Image Analysis Application
Comprehensive microscopy analysis suite replicating Napari Hub plugin functionality
"""

import os
import io
import base64
import json
from pathlib import Path

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

# Web application (if Flask is available)
if HAS_FLASK:
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
    
    # Create uploads directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Global storage
    analysis_cache = {}
    
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
            image = analysis_cache[session_id]['original_image']
            
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
            analysis_cache[session_id]['processed_images'][f'labels_{method}'] = labels
            
            # Generate measurements
            measurements_df = au.calculate_object_statistics(labels, image)
            measurements = measurements_df.to_dict('records')
            analysis_cache[session_id]['results'][f'measurements_{method}'] = measurements
            
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
            image = analysis_cache[session_id]['original_image']
            
            if analysis_type == 'intensity':
                results = au.calculate_intensity_statistics(image)
            elif analysis_type == 'colocalization':
                # Create synthetic second channel for demo
                channel2 = ndi.gaussian_filter(image, sigma=2)
                results = au.calculate_colocalization_coefficients(image, channel2)
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
        """Perform tracking on a real timelapse sequence"""
        return jsonify({'error': 'Tracking not implemented in this version.'}), 501

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
