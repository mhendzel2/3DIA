#!/usr/bin/env python3
"""
Test script to verify Flask batch processing imports work correctly
"""

import sys
import os

sys.path.append('src')

def test_flask_imports():
    """Test that Flask app with batch processing can be imported"""
    try:
        from scientific_analyzer import HAS_BATCH_PROCESSOR, HAS_FLASK
        print(f'✓ Flask available: {HAS_FLASK}')
        print(f'✓ Batch processor available: {HAS_BATCH_PROCESSOR}')
        
        if HAS_BATCH_PROCESSOR:
            from batch_processor import WORKFLOW_TEMPLATES
            print(f'✓ Available workflows: {list(WORKFLOW_TEMPLATES.keys())}')
        
        print('✓ Flask app imports successfully')
        return True
    except Exception as e:
        print(f'✗ Import failed: {e}')
        return False

if __name__ == "__main__":
    success = test_flask_imports()
    sys.exit(0 if success else 1)
