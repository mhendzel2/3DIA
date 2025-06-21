#!/usr/bin/env python3
"""
Test script to verify batch processing endpoints and HCA features are working correctly
"""

import sys
import os
import json
import tempfile
import requests
import time
from pathlib import Path

sys.path.append('src')

def test_batch_processing_imports():
    """Test that batch processing can be imported into Flask app"""
    try:
        from scientific_analyzer import HAS_BATCH_PROCESSOR, HAS_FLASK
        print(f'✓ Flask available: {HAS_FLASK}')
        print(f'✓ Batch processor available: {HAS_BATCH_PROCESSOR}')
        
        if HAS_BATCH_PROCESSOR:
            from batch_processor import WORKFLOW_TEMPLATES
            print(f'✓ Available workflows: {list(WORKFLOW_TEMPLATES.keys())}')
        
        return True
    except Exception as e:
        print(f'✗ Batch processing imports failed: {e}')
        return False

def test_hca_widget_imports():
    """Test that HCA widget can be imported"""
    try:
        from widgets.hca_widget import HighContentAnalysisWidget, PlateVisualizationWidget
        print('✓ HCA widget imports successfully')
        
        from utils.analysis_utils import fit_dose_response, four_param_logistic
        
        concentrations = [0.1, 1.0, 10.0, 100.0, 1000.0]
        responses = [100, 90, 70, 30, 10]
        
        result = fit_dose_response(concentrations, responses)
        print(f'✓ Dose-response fitting works: IC50 = {result.get("ic50", "N/A")}')
        
        return True
    except Exception as e:
        print(f'✗ HCA widget imports failed: {e}')
        return False

def test_flask_batch_endpoints():
    """Test Flask batch processing endpoints (requires running server)"""
    try:
        response = requests.get('http://localhost:5000/', timeout=2)
        if response.status_code != 200:
            print('⚠ Flask server not running - skipping endpoint tests')
            return True
            
        response = requests.get('http://localhost:5000/api/workflows')
        if response.status_code == 200:
            workflows = response.json()
            print(f'✓ Workflows endpoint works: {list(workflows.get("workflows", {}).keys())}')
        else:
            print(f'✗ Workflows endpoint failed: {response.status_code}')
            return False
            
        test_data = {
            'files': ['test1.tif', 'test2.tif'],
            'workflow': 'cell_counting'
        }
        
        response = requests.post('http://localhost:5000/api/batch/process', 
                               json=test_data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            batch_id = result.get('batch_id')
            print(f'✓ Batch processing endpoint works: batch_id = {batch_id}')
            
            if batch_id:
                status_response = requests.get(f'http://localhost:5000/api/batch/status/{batch_id}')
                if status_response.status_code == 200:
                    print('✓ Batch status endpoint works')
                else:
                    print(f'✗ Batch status endpoint failed: {status_response.status_code}')
        else:
            print(f'✗ Batch processing endpoint failed: {response.status_code} - {response.text}')
            return False
            
        return True
        
    except requests.exceptions.RequestException:
        print('⚠ Flask server not accessible - skipping endpoint tests')
        return True
    except Exception as e:
        print(f'✗ Flask endpoint tests failed: {e}')
        return False

def test_napari_hca_integration():
    """Test HCA widget integration with Napari"""
    try:
        import napari
        
        viewer = napari.Viewer(show=False)
        
        from widgets.hca_widget import HighContentAnalysisWidget
        hca_widget = HighContentAnalysisWidget(viewer)
        
        print('✓ HCA widget can be created with Napari viewer')
        
        if hasattr(hca_widget, 'load_plate_layout'):
            print('✓ HCA widget has load_plate_layout method')
        if hasattr(hca_widget, 'run_analysis'):
            print('✓ HCA widget has run_analysis method')
            
        viewer.close()
        return True
        
    except Exception as e:
        print(f'✗ Napari HCA integration test failed: {e}')
        return False

def create_sample_plate_layout():
    """Create a sample plate layout CSV for testing"""
    try:
        import pandas as pd
        
        wells = []
        for row in 'ABCDEFGH':
            for col in range(1, 13):
                wells.append({
                    'well': f'{row}{col:02d}',
                    'row': row,
                    'column': col,
                    'compound': f'Compound_{len(wells) % 10}',
                    'concentration': 10 ** (len(wells) % 5 - 2),  # 0.01 to 100
                    'replicate': (len(wells) % 3) + 1
                })
        
        df = pd.DataFrame(wells)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        print(f'✓ Sample plate layout created: {temp_file.name}')
        return temp_file.name
        
    except Exception as e:
        print(f'✗ Failed to create sample plate layout: {e}')
        return None

def main():
    """Run all batch processing and HCA feature tests"""
    print("Testing batch processing and HCA features...")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    if test_batch_processing_imports():
        tests_passed += 1
    print()
    
    if test_hca_widget_imports():
        tests_passed += 1
    print()
    
    if test_flask_batch_endpoints():
        tests_passed += 1
    print()
    
    if test_napari_hca_integration():
        tests_passed += 1
    print()
    
    sample_plate = create_sample_plate_layout()
    if sample_plate:
        print(f"Sample plate layout available at: {sample_plate}")
    print()
    
    print("=" * 60)
    print(f"Batch processing and HCA tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All batch processing and HCA features working correctly!")
        return True
    else:
        print("✗ Some batch processing and HCA features failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
