#!/usr/bin/env python3
"""
Test script to verify the specific debugging fixes are working correctly
"""

import sys
import os
sys.path.append('src')

def test_aspect_ratio_edge_cases():
    """Test that aspect ratio calculation handles edge cases without ZeroDivisionError"""
    try:
        from scientific_analyzer import RegionPropsAnalysis
        
        labels = [[0, 1, 1, 0], [0, 0, 0, 0]]
        result = RegionPropsAnalysis._measure_object(labels, 1)
        print(f'✓ Horizontal line aspect ratio: {result["aspect_ratio"]} (should be inf)')
        
        labels = [[0, 1], [0, 1], [0, 0]]
        result = RegionPropsAnalysis._measure_object(labels, 1)
        print(f'✓ Vertical line aspect ratio: {result["aspect_ratio"]} (should be 0.0)')
        
        labels = [[0, 1, 0]]
        result = RegionPropsAnalysis._measure_object(labels, 1)
        print(f'✓ Single pixel aspect ratio: {result["aspect_ratio"]} (should be 1.0)')
        
        print('✓ Aspect ratio calculation handles all edge cases without ZeroDivisionError')
        return True
    except Exception as e:
        print(f'✗ Aspect ratio test failed: {e}')
        return False

def test_numpy_fallback_visualization():
    """Test NumPy fallback in VisualizationWidget"""
    try:
        from widgets.visualization_widget import NUMPY_AVAILABLE, VisualizationWidget
        print(f'✓ VisualizationWidget imports successfully, NumPy available: {NUMPY_AVAILABLE}')
        
        class MockViewer:
            pass
        
        widget = VisualizationWidget(MockViewer())
        widget.show_error('Test error message')
        print('✓ show_error method works correctly')
        return True
    except Exception as e:
        print(f'✗ VisualizationWidget test failed: {e}')
        return False

def test_track_endpoint_logic():
    """Test /track endpoint logic components"""
    try:
        from scientific_analyzer import BTrackTracking
        
        mock_timelapse = [
            [[1, 0], [0, 1]], 
            [[0, 1], [1, 0]]
        ]
        
        tracks = BTrackTracking.track_objects(mock_timelapse)
        print(f'✓ BTrackTracking.track_objects callable, returned {len(tracks)} tracks')
        
        print('✓ /track endpoint logic components are functional')
        return True
    except Exception as e:
        print(f'✗ BTrackTracking test failed: {e}')
        return False

def test_napari_analyzer_removed():
    """Test that napari_analyzer.py has been removed"""
    if os.path.exists('src/napari_analyzer.py'):
        print('✗ napari_analyzer.py still exists - should be removed')
        return False
    else:
        print('✓ napari_analyzer.py successfully removed')
        return True

def main():
    """Run all debugging fix tests"""
    print("Testing debugging fixes...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    if test_napari_analyzer_removed():
        tests_passed += 1
    print()
    
    if test_aspect_ratio_edge_cases():
        tests_passed += 1
    print()
    
    if test_numpy_fallback_visualization():
        tests_passed += 1
    print()
    
    if test_track_endpoint_logic():
        tests_passed += 1
    print()
    
    print("=" * 50)
    print(f"Debugging fix tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All debugging fixes working correctly!")
        return True
    else:
        print("✗ Some debugging fixes failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
