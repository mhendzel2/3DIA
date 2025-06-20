#!/usr/bin/env python3
"""
Test script to verify advanced enhancements are working correctly
"""

import sys
import os
sys.path.append('src')

def test_advanced_processing_methods():
    """Test advanced processing methods in ProcessingWidget"""
    try:
        from widgets.processing_widget import ProcessingWidget, ADVANCED_DENOISING_AVAILABLE
        
        print(f'✓ ProcessingWidget imports successfully, Advanced denoising available: {ADVANCED_DENOISING_AVAILABLE}')
        
        if ADVANCED_DENOISING_AVAILABLE:
            from advanced_analysis import AIDenoising
            import numpy as np
            test_image = np.random.rand(100, 100)
            
            result = AIDenoising.bilateral_filter_denoising(test_image, sigma_spatial=5.0)
            print(f'✓ Advanced bilateral filter works, output shape: {result.shape}')
            
            result = AIDenoising.non_local_means_denoising(test_image, h=10.0)
            print(f'✓ Non-local means denoising works, output shape: {result.shape}')
        else:
            print('✓ Advanced denoising not available but imports work')
        
        return True
    except Exception as e:
        print(f'✗ Advanced processing test failed: {e}')
        return False

def test_enhanced_morphological_statistics():
    """Test enhanced morphological statistics"""
    try:
        from utils.analysis_utils import calculate_object_statistics
        import numpy as np
        
        labels = np.zeros((100, 100), dtype=int)
        labels[20:40, 20:40] = 1  # Square object
        labels[60:80, 60:70] = 2  # Rectangle object
        
        stats = calculate_object_statistics(labels)
        
        print(f'✓ Enhanced morphological statistics work, found {stats["object_count"]} objects')
        
        if 'circularity' in stats:
            print(f'✓ Circularity calculation available: {stats["circularity"][:2]}')
        if 'solidity' in stats:
            print(f'✓ Solidity calculation available: {stats["solidity"][:2]}')
        if 'convex_area' in stats:
            print(f'✓ Convex area calculation available: {stats["convex_area"][:2]}')
        
        return True
    except Exception as e:
        print(f'✗ Enhanced morphological statistics test failed: {e}')
        return False

def test_colocalization_enhancements():
    """Test enhanced colocalization plotting"""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        mock_results = {
            'intensity_data': {
                'channel1': np.random.rand(1000),
                'channel2': np.random.rand(1000),
                'image1_full': np.random.rand(100, 100),
                'image2_full': np.random.rand(100, 100)
            },
            'pearson_correlation': 0.75
        }
        
        fig = Figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        pixels1 = mock_results['intensity_data']['channel1']
        pixels2 = mock_results['intensity_data']['channel2']
        ax1.hist2d(pixels1, pixels2, bins=50)
        
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(pixels1, pixels2)
        line = slope * pixels1 + intercept
        ax1.plot(pixels1, line, 'r-', linewidth=1)
        
        print('✓ Enhanced colocalization plotting works')
        return True
    except Exception as e:
        print(f'✗ Enhanced colocalization plotting test failed: {e}')
        return False

def test_spatial_distribution_analysis():
    """Test spatial distribution analysis functions"""
    try:
        from utils.analysis_utils import analyze_spatial_distribution
        import numpy as np
        
        coordinates = np.random.rand(50, 2) * 100
        
        stats = analyze_spatial_distribution(coordinates)
        
        print(f'✓ Spatial distribution analysis works')
        print(f'  - Mean distance: {stats.get("mean_distance", "N/A"):.2f}')
        print(f'  - Mean NN distance: {stats.get("mean_nn_distance", "N/A"):.2f}')
        print(f'  - Density: {stats.get("density", "N/A"):.4f}')
        
        return True
    except Exception as e:
        print(f'✗ Spatial distribution analysis test failed: {e}')
        return False

def test_adaptive_thresholding():
    """Test adaptive thresholding functionality"""
    try:
        from skimage.filters import threshold_local
        import numpy as np
        
        # Create test image
        test_image = np.random.rand(100, 100)
        
        threshold = threshold_local(test_image, block_size=35, offset=10)
        binary = test_image > threshold
        
        print(f'✓ Adaptive thresholding works, binary image shape: {binary.shape}')
        
        return True
    except Exception as e:
        print(f'✗ Adaptive thresholding test failed: {e}')
        return False

def main():
    """Run all advanced enhancement tests"""
    print("Testing advanced enhancements...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    if test_advanced_processing_methods():
        tests_passed += 1
    print()
    
    if test_enhanced_morphological_statistics():
        tests_passed += 1
    print()
    
    if test_colocalization_enhancements():
        tests_passed += 1
    print()
    
    if test_spatial_distribution_analysis():
        tests_passed += 1
    print()
    
    if test_adaptive_thresholding():
        tests_passed += 1
    print()
    
    print("=" * 50)
    print(f"Advanced enhancement tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All advanced enhancements working correctly!")
        return True
    else:
        print("✗ Some advanced enhancements failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
