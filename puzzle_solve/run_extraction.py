#!/usr/bin/env python3
"""
Simple script to run puzzle piece extraction on the terracotta army image.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_puzzle_pieces import extract_puzzle_pieces, extract_with_edge_detection
from advanced_piece_extractor import AdvancedPuzzlePieceExtractor

def main():
    # Path to the terracotta army puzzle image
    image_path = "terracotta_black_bg2/terracotta_army_pieces.png"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        print("Please make sure the terracotta_army_pieces.png file is in the correct location.")
        return
    
    print("=" * 60)
    print("PUZZLE PIECE EXTRACTION FOR TERRACOTTA ARMY IMAGE")
    print("=" * 60)
    
    # Ask user which method to use
    print("\nAvailable extraction methods:")
    print("1. Basic threshold-based extraction")
    print("2. Edge detection-based extraction")
    print("3. Advanced adaptive extraction")
    print("4. Advanced watershed extraction")
    print("5. Advanced color clustering extraction")
    print("6. Run all methods")
    
    choice = input("\nEnter your choice (1-6, or press Enter for option 6): ").strip()
    
    if not choice:
        choice = "6"
    
    output_base = "puzzle_pieces"
    
    try:
        if choice == "1":
            print("\n[Running Basic Threshold-based Extraction]")
            extract_puzzle_pieces(image_path, f"{output_base}/basic_threshold")
            
        elif choice == "2":
            print("\n[Running Edge Detection-based Extraction]")
            extract_with_edge_detection(image_path, f"{output_base}/edge_detection")
            
        elif choice in ["3", "4", "5"]:
            method_map = {
                "3": "adaptive",
                "4": "watershed", 
                "5": "color"
            }
            method = method_map[choice]
            print(f"\n[Running Advanced {method.capitalize()} Extraction]")
            extractor = AdvancedPuzzlePieceExtractor(image_path)
            summary = extractor.extract_pieces(method=method, output_dir=f"{output_base}/advanced_{method}")
            print(f"\nExtracted {summary['total_pieces']} pieces")
            print(f"Results saved to: {summary['output_directory']}")
            
        elif choice == "6":
            print("\n[Running ALL Extraction Methods]")
            print("\nThis will try multiple methods to find the best extraction.")
            print("Results will be saved in separate folders for comparison.\n")
            
            # Run basic methods
            print("1/5: Basic threshold method...")
            extract_puzzle_pieces(image_path, f"{output_base}/basic_threshold")
            
            print("\n2/5: Edge detection method...")
            extract_with_edge_detection(image_path, f"{output_base}/edge_detection")
            
            # Run advanced methods
            extractor = AdvancedPuzzlePieceExtractor(image_path)
            
            print("\n3/5: Advanced adaptive method...")
            summary = extractor.extract_pieces("adaptive", f"{output_base}/advanced_adaptive")
            print(f"   - Found {summary['total_pieces']} pieces")
            
            print("\n4/5: Advanced watershed method...")
            summary = extractor.extract_pieces("watershed", f"{output_base}/advanced_watershed")
            print(f"   - Found {summary['total_pieces']} pieces")
            
            print("\n5/5: Advanced color clustering method...")
            summary = extractor.extract_pieces("color", f"{output_base}/advanced_color")
            print(f"   - Found {summary['total_pieces']} pieces")
            
        else:
            print("Invalid choice. Please run the script again.")
            return
            
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE!")
        print("=" * 60)
        print(f"\nResults saved in: {output_base}/")
        print("\nEach method folder contains:")
        print("  • Individual piece images (PNG with transparency)")
        print("  • Visualization showing all detected pieces")
        print("  • JSON file with piece information (position, size, etc.)")
        print("\nYou can compare the results from different methods to see which")
        print("works best for your puzzle image.")
        
    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease make sure you have OpenCV installed:")
        print("  pip install opencv-python numpy")


if __name__ == "__main__":
    main()