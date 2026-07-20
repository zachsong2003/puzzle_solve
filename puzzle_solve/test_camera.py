#!/usr/bin/env python3
"""
Test script for camera capture functionality
"""

import sys
import os
from camera_capture import CameraPuzzleSolver

def test_basic_functionality():
    """Test basic functionality without camera access."""
    print("🧪 Testing Camera Capture Script")
    print("=" * 50)
    
    try:
        # Test initialization
        solver = CameraPuzzleSolver()
        print("✅ CameraPuzzleSolver initialized successfully")
        
        # Test output directory creation
        if os.path.exists('captured_images'):
            print("✅ Output directory created successfully")
        else:
            print("❌ Output directory not created")
            return False
        
        # Test configuration
        print(f"✅ Server URL configured: {solver.server_url}")
        print(f"✅ Output size configured: {solver.output_size}")
        
        # Test method availability
        methods = ['initialize_camera', 'capture_frame', 'send_to_server', 
                  'save_image_locally', 'frame_to_bytes', 'save_solved_image']
        
        for method in methods:
            if hasattr(solver, method):
                print(f"✅ Method {method} available")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_imports():
    """Test all required imports."""
    print("\n🔍 Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        
        import requests
        print("✅ Requests imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        from PIL import Image
        print("✅ PIL imported successfully")
        
        import base64
        print("✅ Base64 imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Camera Capture Tests\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed!")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎯 READY TO USE!")
    print("=" * 50)
    print("📋 Next steps:")
    print("1. Start the server:")
    print("   uvicorn aiml.puzzle_solve.server:app --host 127.0.0.1 --port 8000")
    print("2. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY=sk-your-key-here")
    print("3. Run the camera capture:")
    print("   python camera_capture.py")
    print("=" * 50)

if __name__ == "__main__":
    main()