#!/usr/bin/env python3
"""
Test script to verify dose-response curve fitting functionality
"""

import sys
sys.path.append('src')

def test_dose_response_fitting():
    """Test dose-response curve fitting with sample data"""
    try:
        from utils.analysis_utils import fit_dose_response, four_param_logistic
        
        concentrations = [0.1, 1.0, 10.0, 100.0, 1000.0]
        responses = [100, 90, 70, 30, 10]  # Decreasing response with increasing concentration
        
        print("Testing dose-response curve fitting...")
        print(f"Concentrations: {concentrations}")
        print(f"Responses: {responses}")
        
        result = fit_dose_response(concentrations, responses)
        
        if 'error' in result:
            print(f"✗ Dose-response fitting failed: {result['error']}")
            return False
        else:
            print(f"✓ Dose-response fitting successful!")
            print(f"  IC50: {result.get('ic50', 'N/A')}")
            print(f"  Method: {result.get('fit_method', 'N/A')}")
            print(f"  Min response: {result.get('min_response', 'N/A')}")
            print(f"  Max response: {result.get('max_response', 'N/A')}")
            
            short_conc = [1.0, 10.0]
            short_resp = [90, 30]
            
            result2 = fit_dose_response(short_conc, short_resp)
            if 'error' in result2:
                print("✓ Correctly handles insufficient data")
            else:
                print("⚠ Should have failed with insufficient data")
            
            return True
            
    except Exception as e:
        print(f"✗ Dose-response test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_dose_response_fitting()
    sys.exit(0 if success else 1)
