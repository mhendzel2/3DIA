"""
PyMaris Scientific Image Analyzer
Napari-based microscopy analysis suite with enhanced web interface
Dual-interface approach: Napari desktop + web application
"""

import sys
import os

def main():
    """Main entry point for the scientific image analyzer"""
    print("PyMaris Scientific Image Analyzer")
    print("=" * 60)
    print("Dual Interface Architecture:")
    print("• Napari Desktop Application (Primary Platform)")
    print("• Enhanced Web Interface (Complementary Access)")
    print("=" * 60)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'napari':
            launch_napari_interface()
        elif mode == 'web':
            launch_web_interface()
        elif mode == 'enhanced':
            launch_enhanced_web_interface()
        else:
            print(f"Unknown mode: {mode}")
            show_usage()
    else:
        # Default: try enhanced web interface (most reliable)
        launch_enhanced_web_interface()

def launch_napari_interface():
    """Launch the Napari-based desktop application"""
    print("\nLaunching Napari Desktop Application...")
    try:
        from main_napari import main as napari_main
        napari_main()
        return True
    except ImportError as e:
        print(f"Napari interface not available: {e}")
        print("Note: Napari requires additional dependencies")
        print("Install with: pip install -e \".[napari]\"")
        return False

def launch_web_interface():
    """Launch the basic web interface"""
    print("\nLaunching Basic Web Interface...")
    try:
        from simple_analyzer import run_server
        run_server(port=5000)
        return True
    except ImportError as e:
        print(f"Basic web interface not available: {e}")
        return False

def launch_enhanced_web_interface():
    """Launch the enhanced web interface with Flask"""
    print("\nLaunching Enhanced Web Interface...")
    try:
        from scientific_analyzer import app
        if app:
            print("Starting Flask application...")
            app.run(host='0.0.0.0', port=5000, debug=True)
            return True
        else:
            print("Flask application not available")
            return False
    except ImportError as e:
        print(f"Enhanced web interface not available: {e}")
        print("Falling back to simple analyzer...")
        try:
            from simple_analyzer import run_server
            run_server(port=5000)
            return True
        except ImportError as e2:
            print(f"No interface available: {e2}")
            return False

def show_usage():
    """Show usage information"""
    print("\nUsage:")
    print("  python main.py            # Launch enhanced web interface")
    print("  python main.py napari     # Launch Napari desktop application")
    print("  python main.py web        # Launch basic web interface")
    print("  python main.py enhanced   # Launch enhanced web interface")
    print("\nRecommended:")
    print("  Use 'napari' for full functionality with desktop UI")
    print("  Use 'enhanced' for web-based access with advanced features")

if __name__ == "__main__":
    main()
