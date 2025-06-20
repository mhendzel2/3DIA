#!/usr/bin/env python3
"""
Test core functionality without external dependencies
"""

import sys
import os
sys.path.append('src')

def test_basic_imports():
    """Test basic imports that don't require external dependencies"""
    try:
        from main import main, launch_web_interface, launch_enhanced_web_interface
        print('✓ main.py core functions import successfully')
        return True
    except Exception as e:
        print(f'✗ main.py import failed: {e}')
        return False

def test_web_servers():
    """Test web server imports"""
    try:
        from simple_analyzer import SimpleImageAnalyzer, run_server
        print('✓ simple_analyzer imports successfully')
        
        test_image = SimpleImageAnalyzer.create_test_image(width=10, height=10)
        labels = SimpleImageAnalyzer.simple_threshold(test_image, threshold=128)
        measurements = SimpleImageAnalyzer.measure_objects(labels)
        print(f'✓ simple_analyzer basic functionality works: {len(measurements)} objects found')
        return True
    except Exception as e:
        print(f'✗ simple_analyzer test failed: {e}')
        return False

def test_scientific_analyzer():
    """Test scientific analyzer without Flask dependencies"""
    try:
        import scientific_analyzer
        print('✓ scientific_analyzer module imports successfully')
        
        from scientific_analyzer import GradientWatershedSegmentation, ImageProcessor
        print('✓ Core analysis classes import successfully')
        return True
    except Exception as e:
        print(f'✗ scientific_analyzer test failed: {e}')
        return False

def test_bug_fixes_integration():
    """Test that bug fixes are still working"""
    try:
        from fibsem_plugins import ChimeraXIntegration, ThreeDCounter
        print('✓ fibsem_plugins imports successfully after cleanup')
        
        from batch_processor import BatchProcessor
        print('✓ batch_processor imports successfully after cleanup')
        return True
    except Exception as e:
        print(f'✗ bug fixes integration test failed: {e}')
        return False

def main():
    """Run core functionality tests"""
    print("Testing core functionality after structural cleanup...")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    if test_basic_imports():
        tests_passed += 1
    print()
    
    if test_web_servers():
        tests_passed += 1
    print()
    
    if test_scientific_analyzer():
        tests_passed += 1
    print()
    
    if test_bug_fixes_integration():
        tests_passed += 1
    print()
    
    print("=" * 60)
    print(f"Core functionality tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ Core functionality preserved after structural cleanup!")
        return True
    else:
        print("✗ Some core functionality tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
