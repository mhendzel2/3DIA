#!/usr/bin/env python3
"""
Simple import test to verify the bug fixes don't break basic functionality
"""

import sys
import os
sys.path.append('src')

def test_basic_imports():
    """Test that all modules can be imported without errors"""
    try:
        print("Testing fibsem_plugins imports...")
        from fibsem_plugins import (
            ChimeraXIntegration, 
            ThreeDCounter, 
            TomoSliceAnalyzer,
            AcceleratedClassification,
            MembraneSegmenter,
            FIBSEMAnalyzer
        )
        print("✓ All fibsem_plugins classes imported successfully")
        
        print("Testing batch_processor imports...")
        from batch_processor import BatchProcessor, WORKFLOW_TEMPLATES
        print("✓ BatchProcessor and templates imported successfully")
        
        print("Testing bug_fixes imports...")
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from bug_fixes import ChimeraXPathFix, MRCExportFix, TIFFExportFix
        print("✓ All bug_fixes classes imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_instantiation():
    """Test that classes can be instantiated without system calls"""
    try:
        from fibsem_plugins import ThreeDCounter, TomoSliceAnalyzer
        from batch_processor import BatchProcessor
        
        counter = ThreeDCounter()
        print("✓ ThreeDCounter instantiated")
        
        analyzer = TomoSliceAnalyzer()
        print("✓ TomoSliceAnalyzer instantiated")
        
        processor = BatchProcessor()
        print("✓ BatchProcessor instantiated")
        
        return True
        
    except Exception as e:
        print(f"✗ Instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing basic imports and instantiation...")
    print("=" * 50)
    
    success = True
    
    if not test_basic_imports():
        success = False
    print()
    
    if not test_basic_instantiation():
        success = False
    
    print("=" * 50)
    if success:
        print("✓ All basic tests passed!")
    else:
        print("✗ Some tests failed")
    
    sys.exit(0 if success else 1)
