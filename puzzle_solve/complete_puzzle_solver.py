#!/usr/bin/env python3
"""
Complete puzzle solver pipeline:
1. Extract pieces from scrambled image using edge detection
2. Map pieces to their row/col positions
3. Reconstruct the puzzle with connected edges
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import argparse

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piece_position_mapper import PuzzlePiecePositionMapper
from puzzle_reconstructor import PuzzleReconstructor

def select_image_from_folder():
    """Allow user to select an image from a folder."""
    print("\n" + "=" * 50)
    print("SELECT IMAGE FROM FOLDER")
    print("=" * 50)
    
    # Get folder path
    folder_path = input("Enter the folder path (or press Enter for default): ").strip()
    if not folder_path:
        # folder_path = "terracotta_black_bg2"
        folder_path = "terracotta_black_bg2"
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return None
    
    # List image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return None
    
    # Display available images
    print(f"\nAvailable images in {folder_path}:")
    for i, img_path in enumerate(image_files, 1):
        print(f"{i}. {img_path.name}")
    
    # Get user selection
    while True:
        try:
            choice = input(f"\nSelect image (1-{len(image_files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(image_files):
                selected_path = str(image_files[idx])
                print(f"Selected: {image_files[idx].name}")
                return selected_path
            else:
                print("Invalid selection. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number.")

def detect_white_paper(image):
    """
    Detect white paper rectangle in the image using color-based detection.
    White paper on grey background.
    
    Args:
        image: Input image
    
    Returns:
        Contour of the detected paper or None if not found
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Also work with grayscale for brightness detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Method 1: Direct brightness thresholding for white paper
    # White paper should be brighter than grey background
    # Typical white paper: 200-255, Grey background: 100-150
    _, white_mask = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY)
    
    # # Method 2: HSV-based detection for white/light colors
    # # White has low saturation and high value
    # lower_white = np.array([0, 0, 180])  # Low saturation, high value
    # upper_white = np.array([180, 30, 255])  # Any hue, low saturation, high value
    # hsv_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # # Combine both masks
    # combined_mask = cv2.bitwise_or(white_mask, hsv_mask)
    
    # Clean up the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    
    # Remove small noise
    combined_mask = white_mask
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilate to ensure edges are connected
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Look for the largest rectangular white region
    for contour in contours[:3]:  # Check top 3 largest contours
        area = cv2.contourArea(contour)
        
        # Check if it's large enough (at least 15% of image area)
        if area < (image.shape[0] * image.shape[1] * 0.15):
            continue
        
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Check if it's roughly rectangular (4-6 vertices to allow some flexibility)
        if 4 <= len(approx) <= 6:
            # If it has more than 4 points, try to simplify to 4
            if len(approx) > 4:
                # Use convex hull and then approximate again
                hull = cv2.convexHull(contour)
                hull_perimeter = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * hull_perimeter, True)
                
                # If still not 4 points, find the 4 corner points
                if len(approx) != 4:
                    # Get bounding rectangle
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    approx = np.int32(box).reshape(-1, 1, 2)
            
            return approx
    
    # If no good quadrilateral found, try to use the largest white region's bounding box
    if contours:
        largest_contour = contours[0]
        area = cv2.contourArea(largest_contour)
        
        # If the largest region is significant
        if area > (image.shape[0] * image.shape[1] * 0.1):
            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box).reshape(-1, 1, 2)
            return box
    
    return None

def detect_white_paper_advanced(image):
    """
    Alternative detection method using color difference between white paper and grey background.
    
    Args:
        image: Input image
    
    Returns:
        Contour of the detected paper or None if not found
    """
    # Convert to LAB color space for better color separation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract L channel (lightness)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(l_channel)
    
    # Use Otsu's method to find optimal threshold between white and grey
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Check if it's large enough
    if area > (image.shape[0] * image.shape[1] * 0.1):
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box).reshape(-1, 1, 2)
        return box
    
    return None

def crop_to_paper(image, paper_contour):
    """
    Crop and perspective-correct the image to just the paper region.
    
    Args:
        image: Input image
        paper_contour: Contour of the paper (4 points)
    
    Returns:
        Cropped and perspective-corrected image
    """
    # Reshape contour points
    pts = paper_contour.reshape(4, 2)
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and diff to find corners
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
    
    # Compute width and height of new image
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply perspective transform
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def capture_from_camera(args):
    """Capture an image from camera with automatic white paper detection and cropping."""
    print("\n" + "=" * 50)
    print("CAPTURE IMAGE FROM CAMERA")
    print("=" * 50)
    
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("\nCamera opened successfully!")
    print("Instructions:")
    print("- Position the white paper with puzzle pieces in view")
    print("- Press SPACE to capture image")
    print("- Press 'q' to quit without capturing")
    print("- Press 'r' to retry capture")
    
    captured_image = None
    window_name = "Puzzle Capture - Position white paper and press SPACE"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera")
            break
        
        # Try to detect paper in real-time for preview
        display_frame = frame.copy()
        paper_contour = detect_white_paper(frame)
        
        if paper_contour is not None:
            # Draw the detected paper outline
            cv2.drawContours(display_frame, [paper_contour], -1, (0, 255, 0), 3)
            cv2.putText(display_frame, "Paper detected!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Position white paper in view", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key - capture
            captured_image = frame.copy()
            print("\nImage captured!")
            
            # Try multiple detection methods
            print("Detecting white paper...")
            paper_contour = detect_white_paper(captured_image)
            if paper_contour is None:
                print("White paper not detected, continue")
                continue

            break
            
            # If first method fails, try advanced method
            if paper_contour is None:
                print("Trying advanced detection method...")
                paper_contour = detect_white_paper_advanced(captured_image)
            
            if paper_contour is not None:
                print("White paper detected! Cropping to paper region...")
                
                # Show detected paper
                preview = captured_image.copy()
                cv2.drawContours(preview, [paper_contour], -1, (0, 255, 0), 3)
                cv2.imshow("Detected Paper - Press 'c' to crop, 'r' to retry", preview)
                
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('c'):  # Crop
                        # Crop to paper region
                        cropped_image = crop_to_paper(captured_image, paper_contour)
                        
                        # Show cropped result
                        cv2.imshow("Cropped Result - Press 's' to save, 'r' to retry", cropped_image)
                        
                        while True:
                            key3 = cv2.waitKey(0) & 0xFF
                            if key3 == ord('s'):  # Save
                                # Save the cropped image
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_path = f"{args.output_dir}/puzzle_capture_{timestamp}_cropped.jpg"
                                
                                # Create directory if it doesn't exist
                                Path("puzzle_solve/captured_images").mkdir(parents=True, exist_ok=True)
                                
                                cv2.imwrite(save_path, cropped_image)
                                print(f"Cropped image saved to: {save_path}")
                                
                                # Also save original for reference
                                orig_path = f"{args.output_dir}/puzzle_capture_{timestamp}_original.jpg"
                                cv2.imwrite(orig_path, captured_image)
                                print(f"Original image saved to: {orig_path}")
                                
                                cap.release()
                                cv2.destroyAllWindows()
                                return save_path
                                
                            elif key3 == ord('r'):  # Retry
                                print("Retrying capture...")
                                break
                        
                        if key3 == ord('r'):
                            break
                            
                    elif key2 == ord('r'):  # Retry
                        print("Retrying capture...")
                        break
                    elif key2 == ord('q'):  # Quit
                        cap.release()
                        cv2.destroyAllWindows()
                        return None
            else:
                print("Warning: Could not detect white paper in image.")
                print("Options:")
                print("  's' - Save anyway (without cropping)")
                print("  'r' - Retry capture")
                print("  'q' - Quit")
                
                cv2.imshow("No Paper Detected - Press 's' to save anyway, 'r' to retry", captured_image)
                
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('s'):  # Save anyway
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = f"puzzle_solve/captured_images/puzzle_capture_{timestamp}.jpg"
                        
                        Path("puzzle_solve/captured_images").mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(save_path, captured_image)
                        print(f"Image saved (uncropped) to: {save_path}")
                        
                        cap.release()
                        cv2.destroyAllWindows()
                        return save_path
                        
                    elif key2 == ord('r'):  # Retry
                        print("Retrying capture...")
                        break
                    elif key2 == ord('q'):  # Quit
                        cap.release()
                        cv2.destroyAllWindows()
                        return None
                
                if key2 == ord('r'):
                    continue
        
        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_image is None or paper_contour is None:
        return None

    cropped_image = crop_to_paper(captured_image, paper_contour)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{args.output_dir}/puzzle_capture_{timestamp}_cropped.jpg"
    cv2.imwrite(save_path, cropped_image)
    print(f"Cropped image saved to: {save_path}")
    # Also save original for reference
    orig_path = f"{args.output_dir}/puzzle_capture_{timestamp}_original.jpg"
    cv2.imwrite(orig_path, captured_image)
    print(f"Original image saved to: {orig_path}")

    return save_path

def get_scrambled_image(args):
    """Get scrambled image from user's choice."""
    # print("\n" + "=" * 50)
    # print("IMAGE INPUT METHOD")
    # print("=" * 50)
    # print("1. Use default image (terracotta_army_pieces.png)")
    # print("2. Select from folder")
    # print("3. Capture from camera")
    
    # choice = input("\nSelect input method (1-3): ").strip()
    choice = "3"
    
    if choice == "1":
        return "terracotta_black_bg2/terracotta_army_pieces.png"
    elif choice == "2":
        return select_image_from_folder()
    elif choice == "3":
        return capture_from_camera(args)
    else:
        print("Invalid choice. Using default image.")
        return "terracotta_black_bg2/terracotta_army_pieces.png"

def get_reference_directory():
    """Get reference directory from user."""
    print("\n" + "=" * 50)
    print("REFERENCE DIRECTORY")
    print("=" * 50)
    
    # ref_dir = input("Enter reference directory path (or press Enter for default): ").strip()
    # if not ref_dir:
    ref_dir = "terracotta_black_bg2"
    
    if not Path(ref_dir).exists():
        print(f"Warning: Directory does not exist: {ref_dir}")
        create = input("Create directory? (y/n): ").strip().lower()
        if create == 'y':
            Path(ref_dir).mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {ref_dir}")
        else:
            print("Using default directory.")
            ref_dir = "terracotta_black_bg2"
    
    return ref_dir

def main(args):
    print("=" * 70)
    print("COMPLETE PUZZLE SOLVING PIPELINE")
    print("=" * 70)
    
    # Get scrambled image
    scrambled_image = get_scrambled_image(args)
    if not scrambled_image:
        print("No image selected. Exiting.")
        return
    
    # Verify image exists
    if not Path(scrambled_image).exists():
        print(f"Error: Image does not exist: {scrambled_image}")
        return
    
    # Get reference directory
    reference_dir = get_reference_directory()
    
    # Set output directory
    # output_dir = "puzzle_solve/mapped_pieces"
    output_dir = args.output_dir
    
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Scrambled image: {scrambled_image}")
    print(f"Reference directory: {reference_dir}")
    print(f"Output directory: {output_dir}")
    
    # Ask user to confirm
    # proceed = input("\nProceed with these settings? (y/n): ").strip().lower()
    # if proceed != 'y':
    #     print("Operation cancelled.")
    #     return
    
    try:
        # Step 1: Extract and map pieces
        print("\n" + "=" * 70)
        print("STEP 1: EXTRACTING AND MAPPING PIECES")
        print("-" * 50)
        
        mapper = PuzzlePiecePositionMapper(reference_dir, scrambled_image)
        mapping_info = mapper.map_all_pieces(output_dir)
        
        if not mapping_info or mapping_info['mapped_pieces'] == 0:
            print("Error: No pieces were successfully mapped")
            print("\nThis might happen if:")
            print("1. No reference pieces (row*_col*.png) found in reference directory")
            print("2. The pieces couldn't be extracted from the scrambled image")
            print("3. The pieces couldn't be matched to references")
            
            # Ask if user wants to just extract pieces
            extract_only = input("\nDo you want to extract pieces without mapping? (y/n): ").strip().lower()
            if extract_only == 'y':
                print("\nExtracting pieces without position mapping...")
                from extract_puzzle_pieces import extract_puzzle_pieces
                extract_puzzle_pieces(scrambled_image, output_dir + "/extracted_only")
                print(f"Pieces extracted to: {output_dir}/extracted_only")
            return
        
        print(f"\n✓ Successfully mapped {mapping_info['mapped_pieces']} pieces")
        
        # Step 2: Reconstruct puzzle with edge matching
        print("\n" + "=" * 70)
        print("STEP 2: RECONSTRUCTING PUZZLE WITH EDGE MATCHING")
        print("-" * 50)
        
        reconstructor = PuzzleReconstructor(output_dir)
        
        # Create edge-matched reconstruction
        print("\nCreating edge-matched reconstruction...")
        reconstructor.reconstruct_with_edge_matching()
        
        # Create seamless reconstruction
        print("\nCreating seamless blended reconstruction...")
        reconstructor.create_seamless_reconstruction()
        
        # Summary
        print("\n" + "=" * 70)
        print("PUZZLE SOLVING COMPLETE!")
        print("=" * 70)
        print("\nGenerated files in", output_dir + ":")
        print("\n1. Individual pieces:")
        print("   - piece_r*_c*_score*.png: Individual pieces with position labels")
        print("\n2. Visualizations:")
        print("   - mapped_visualization.jpg: Shows detected pieces with labels")
        print("   - reconstructed_grid.jpg: Basic grid reconstruction")
        print("   - reconstructed_grid_connected.jpg: Edge-matched reconstruction")
        print("   - reconstructed_grid_with_labels.jpg: Reconstruction with grid labels")
        print("   - reconstructed_seamless.jpg: Seamless blended reconstruction")
        print("\n3. Data files:")
        print("   - piece_mappings.json: Detailed mapping information")
        print("\n4. Debug files:")
        print("   - debug/: Contains edge detection intermediate images")
        
        # Show statistics
        if mapping_info['mappings']:
            import numpy as np
            scores = [m['match_score'] for m in mapping_info['mappings']]
            print("\n" + "=" * 70)
            print("STATISTICS")
            print("-" * 50)
            print(f"Total pieces in scrambled image: {mapping_info['total_pieces']}")
            print(f"Successfully mapped: {mapping_info['mapped_pieces']}")
            print(f"Reference pieces available: {mapping_info['reference_pieces']}")
            print(f"\nMatching confidence:")
            print(f"  Average score: {np.mean(scores):.3f}")
            print(f"  Min score: {np.min(scores):.3f}")
            print(f"  Max score: {np.max(scores):.3f}")
            
            high_conf = sum(1 for s in scores if s > 0.5)
            med_conf = sum(1 for s in scores if 0.3 <= s <= 0.5)
            low_conf = sum(1 for s in scores if s < 0.3)
            
            print(f"\nConfidence distribution:")
            print(f"  High (>0.5): {high_conf} pieces")
            print(f"  Medium (0.3-0.5): {med_conf} pieces")
            print(f"  Low (<0.3): {low_conf} pieces")
        
        print("\n✅ All operations completed successfully!")
        
        # # Ask if user wants to view results
        # view = input("\nDo you want to view the reconstructed puzzle? (y/n): ").strip().lower()
        # if view == 'y':
        #     result_path = Path(output_dir) / "reconstructed_seamless.jpg"
        #     if result_path.exists():
        #         img = cv2.imread(str(result_path))
        #         if img is not None:
        #             cv2.imshow("Reconstructed Puzzle - Press any key to close", img)
        #             cv2.waitKey(0)
        #             cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n❌ Error during puzzle solving: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("1. The scrambled image exists at the specified path")
        print("2. Reference pieces (row*_col*.png) exist in the reference directory")
        print("3. OpenCV is installed: pip install opencv-python numpy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="puzzle_solve/mapped_pieces", required=False)
    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)