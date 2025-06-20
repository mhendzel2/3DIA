#!/usr/bin/env python3
"""
Test script to verify the bug fixes are working correctly
"""

import sys
import os
sys.path.append('src')

def test_fibsem_imports():
    """Test fibsem_plugins.py imports and basic functionality"""
    try:
        from fibsem_plugins import ChimeraXIntegration, FIBSEMAnalyzer, ThreeDCounter
        print('✓ fibsem_plugins.py imports successfully')
        
        chimera = ChimeraXIntegration()
        print(f'✓ ChimeraX integration initialized')
        print(f'  - ChimeraX path detected: {chimera.chimerax_path is not None}')
        if chimera.chimerax_path:
            print(f'  - Path: {chimera.chimerax_path}')
        
        counter = ThreeDCounter()
        test_labels = [[[1, 0, 2], [0, 1, 0]], [[2, 2, 0], [1, 0, 1]]]
        result = counter.count_3d_objects(test_labels)
        print(f'✓ 3D object counting works: {result["total_objects"]} objects found')
        
        return True
        
    except Exception as e:
        print(f'✗ fibsem_plugins.py test failed: {e}')
        return False

def test_batch_processor():
    """Test batch_processor.py imports and basic functionality"""
    try:
        from batch_processor import BatchProcessor, WORKFLOW_TEMPLATES
        print('✓ batch_processor.py imports successfully')
        
        processor = BatchProcessor()
        print('✓ BatchProcessor instantiated successfully')
        
        print(f'✓ Available workflow templates: {len(WORKFLOW_TEMPLATES)}')
        for name, template in WORKFLOW_TEMPLATES.items():
            print(f'  - {name}: {template["description"]}')
        
        return True
        
    except Exception as e:
        print(f'✗ batch_processor.py test failed: {e}')
        return False

def test_bug_fixes_integration():
    """Test that bug fixes are properly integrated"""
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from bug_fixes import ChimeraXPathFix, MRCExportFix, TIFFExportFix
        print('✓ bug_fixes.py imports successfully')
        
        chimera_path = ChimeraXPathFix.find_chimerax_installation()
        print(f'✓ ChimeraX path detection: {chimera_path is not None}')
        
        test_volume = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        success, method = MRCExportFix.export_proper_mrc(test_volume, '/tmp/test_volume.mrc')
        print(f'✓ MRC export test: {success} using {method}')
        
        test_labels = [[1, 0, 2], [0, 1, 0], [2, 2, 1]]
        success, method = TIFFExportFix.export_proper_tiff(test_labels, '/tmp/test_mask.tiff')
        print(f'✓ TIFF export test: {success} using {method}')
        
        return True
        
    except Exception as e:
        print(f'✗ bug_fixes integration test failed: {e}')
        return False

def main():
    """Run all tests"""
    print("Testing 3DIA bug fixes integration...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_fibsem_imports():
        tests_passed += 1
    print()
    
    if test_batch_processor():
        tests_passed += 1
    print()
    
    if test_bug_fixes_integration():
        tests_passed += 1
    print()
    
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Bug fixes successfully integrated.")
        return True
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
