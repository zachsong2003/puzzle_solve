#!/usr/bin/env python3
"""
Check for reference pieces and run the mapping process.
"""

import os
from pathlib import Path
import cv2

def check_reference_pieces(directory):
    """Check what reference piece files exist in the directory."""
    ref_dir = Path(directory)
    
    if not ref_dir.exists():
        print(f"Error: Directory does not exist: {directory}")
        return []
    
    # Look for row*_col* pattern files
    piece_files = list(ref_dir.glob("row*_col*.png"))
    
    if not piece_files:
        print(f"No reference pieces (row*_col*.png) found in {directory}")
        print("\nChecking what files exist in the directory:")
        all_files = list(ref_dir.glob("*.png"))
        for f in all_files[:20]:  # Show first 20 files
            print(f"  - {f.name}")
        if len(all_files) > 20:
            print(f"  ... and {len(all_files) - 20} more files")
    else:
        print(f"Found {len(piece_files)} reference piece files:")
        for f in sorted(piece_files):
            print(f"  - {f.name}")
    
    return piece_files

def check_scrambled_image(image_path):
    """Check if the scrambled image exists and can be loaded."""
    img_path = Path(image_path)
    
    if not img_path.exists():
        print(f"Error: Scrambled image does not exist: {image_path}")
        return False
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return False
    
    print(f"Scrambled image loaded successfully: {img.shape[1]}x{img.shape[0]} pixels")
    return True

def main():
    # Paths
    reference_dir = "terracotta_black_bg2"
    scrambled_image = "terracotta_black_bg2/terracotta_army_pieces.png"
    
    print("=" * 60)
    print("CHECKING PUZZLE FILES")
    print("=" * 60)
    print()
    
    # Check reference pieces
    print("1. Checking reference pieces...")
    print("-" * 40)
    ref_pieces = check_reference_pieces(reference_dir)
    print()
    
    # Check scrambled image
    print("2. Checking scrambled image...")
    print("-" * 40)
    has_scrambled = check_scrambled_image(scrambled_image)
    print()
    
    # If we have what we need, run the mapping
    if ref_pieces and has_scrambled:
        print("=" * 60)
        print("RUNNING PIECE POSITION MAPPING")
        print("=" * 60)
        print()
        
        from piece_position_mapper import PuzzlePiecePositionMapper
        
        try:
            mapper = PuzzlePiecePositionMapper(reference_dir, scrambled_image)
            mapping_info = mapper.map_all_pieces("puzzle_solve/mapped_pieces")
            
            if mapping_info and mapping_info['mapped_pieces'] > 0:
                print("\n✓ Mapping completed successfully!")
            else:
                print("\n⚠ Mapping completed but no pieces were mapped.")
                
        except Exception as e:
            print(f"\nError during mapping: {e}")
            import traceback
            traceback.print_exc()
            
    elif not ref_pieces and has_scrambled:
        print("=" * 60)
        print("ALTERNATIVE: EXTRACTING PIECES WITHOUT POSITION MAPPING")
        print("=" * 60)
        print("\nSince there are no reference pieces (row*_col*.png files),")
        print("we'll just extract the pieces from the scrambled image.")
        print()
        
        response = input("Would you like to extract pieces anyway? (y/n): ").strip().lower()
        
        if response == 'y':
            from extract_puzzle_pieces import extract_puzzle_pieces
            from advanced_piece_extractor import AdvancedPuzzlePieceExtractor
            
            print("\nExtracting pieces using multiple methods...")
            
            # Method 1: Basic extraction
            print("\n1. Basic threshold method...")
            extract_puzzle_pieces(scrambled_image, "puzzle_solve/puzzle_pieces/basic_threshold")
            
            # Method 2: Advanced extraction
            print("\n2. Advanced adaptive method...")
            extractor = AdvancedPuzzlePieceExtractor(scrambled_image)
            summary = extractor.extract_pieces("adaptive", "puzzle_solve/puzzle_pieces/advanced_adaptive")
            print(f"   Extracted {summary['total_pieces']} pieces")
            
            print("\n✓ Extraction completed!")
            print("Check the results in puzzle_solve/puzzle_pieces/")
    else:
        print("⚠ Cannot proceed without the scrambled image.")


if __name__ == "__main__":
    main()