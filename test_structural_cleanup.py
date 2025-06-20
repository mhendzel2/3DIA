#!/usr/bin/env python3
"""
Test script to verify structural cleanup is working correctly
"""

import sys
import os
sys.path.append('src')

def test_main_imports():
    """Test that main.py imports work correctly"""
    try:
        from main import main, launch_napari_interface, launch_web_interface, launch_enhanced_web_interface
        print('✓ main.py imports successfully')
        return True
    except Exception as e:
        print(f'✗ main.py import failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_napari_imports():
    """Test that main_napari.py imports work correctly"""
    try:
        from main_napari import main
        print('✓ main_napari.py imports successfully')
        return True
    except Exception as e:
        print(f'✗ main_napari.py import failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_widget_imports():
    """Test that advanced widgets import correctly"""
    try:
        from widgets.analysis_widget import AnalysisWidget
        print('✓ AnalysisWidget imports successfully')
    except Exception as e:
        print(f'✗ AnalysisWidget import failed: {e}')
        return False
        
    try:
        from widgets.processing_widget import ProcessingWidget
        print('✓ ProcessingWidget imports successfully')
    except Exception as e:
        print(f'✗ ProcessingWidget import failed: {e}')
        return False
        
    try:
        from widgets.segmentation_widget import SegmentationWidget
        print('✓ SegmentationWidget imports successfully')
    except Exception as e:
        print(f'✗ SegmentationWidget import failed: {e}')
        return False
        
    try:
        from widgets.visualization_widget import VisualizationWidget
        print('✓ VisualizationWidget imports successfully')
    except Exception as e:
        print(f'✗ VisualizationWidget import failed: {e}')
        return False
        
    try:
        from widgets.file_io_widget import FileIOWidget
        print('✓ FileIOWidget imports successfully')
        return True
    except Exception as e:
        print(f'✗ FileIOWidget import failed: {e}')
        return False

def test_web_server_imports():
    """Test that web server imports work correctly"""
    try:
        from scientific_analyzer import app
        print('✓ scientific_analyzer Flask app imports successfully')
    except Exception as e:
        print(f'✗ scientific_analyzer import failed: {e}')
        return False
        
    try:
        from simple_analyzer import run_server
        print('✓ simple_analyzer imports successfully')
        return True
    except Exception as e:
        print(f'✗ simple_analyzer import failed: {e}')
        return False

def test_removed_files():
    """Test that removed files are actually gone"""
    removed_files = [
        'src/enhanced_web_app.py',
        'src/analysis_widget.py', 
        'src/processing_widget.py',
        'src/segmentation_widget.py',
        'src/dist/'
    ]
    
    all_removed = True
    for file_path in removed_files:
        if os.path.exists(file_path):
            print(f'✗ File should be removed but still exists: {file_path}')
            all_removed = False
        else:
            print(f'✓ File correctly removed: {file_path}')
    
    return all_removed

def main():
    """Run all structural cleanup tests"""
    print("Testing structural cleanup...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    if test_main_imports():
        tests_passed += 1
    print()
    
    if test_napari_imports():
        tests_passed += 1
    print()
    
    if test_widget_imports():
        tests_passed += 1
    print()
    
    if test_web_server_imports():
        tests_passed += 1
    print()
    
    if test_removed_files():
        tests_passed += 1
    print()
    
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All structural cleanup tests passed!")
        return True
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
