import cv2
import numpy as np
import os
from pathlib import Path

def extract_puzzle_pieces(image_path, output_dir="extracted_pieces"):
    """
    Extract individual puzzle pieces from a scrambled puzzle image.
    
    Args:
        image_path: Path to the input image containing scrambled puzzle pieces
        output_dir: Directory to save extracted pieces
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read the image
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Create a copy for visualization
    img_display = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive threshold to handle varying lighting conditions
    # This is better for puzzle pieces with varying colors
    thresh = cv2.adaptiveThreshold(filtered, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    
    # Close small gaps in piece boundaries
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} initial contours")
    
    # Filter contours based on area to remove noise
    min_area = 500  # Minimum area for a valid puzzle piece
    max_area = img.shape[0] * img.shape[1] * 0.5  # Maximum area (half of image)
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
    
    print(f"Found {len(valid_contours)} valid puzzle pieces")
    
    # Extract and save each piece
    pieces_info = []
    
    for idx, contour in enumerate(valid_contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding around the piece
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        # Extract the piece region
        piece = img[y_start:y_end, x_start:x_end]
        
        # Create a mask for the piece
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        piece_mask = mask[y_start:y_end, x_start:x_end]
        
        # Apply mask to get only the piece with transparent background
        piece_rgba = cv2.cvtColor(piece, cv2.COLOR_BGR2BGRA)
        piece_rgba[:, :, 3] = piece_mask
        
        # Save the piece
        piece_filename = f"piece_{idx:03d}.png"
        piece_path = output_path / piece_filename
        cv2.imwrite(str(piece_path), piece_rgba)
        
        # Store piece information
        pieces_info.append({
            'index': idx,
            'filename': piece_filename,
            'position': (x, y),
            'size': (w, h),
            'area': cv2.contourArea(contour),
            'perimeter': cv2.arcLength(contour, True)
        })
        
        # Draw contour on display image
        color = np.random.randint(0, 255, 3).tolist()
        cv2.drawContours(img_display, [contour], -1, color, 2)
        cv2.putText(img_display, str(idx), (x + w//2, y + h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save visualization
    visualization_path = output_path / "pieces_visualization.jpg"
    cv2.imwrite(str(visualization_path), img_display)
    
    # Save debug images
    debug_path = output_path / "debug"
    debug_path.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_path / "threshold.jpg"), thresh)
    cv2.imwrite(str(debug_path / "morphology.jpg"), opened)
    
    # Save pieces information
    import json
    info_path = output_path / "pieces_info.json"
    with open(info_path, 'w') as f:
        json.dump(pieces_info, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"- Extracted {len(valid_contours)} pieces")
    print(f"- Pieces saved to: {output_path}")
    print(f"- Visualization saved to: {visualization_path}")
    print(f"- Pieces info saved to: {info_path}")
    
    return pieces_info


def extract_with_edge_detection(image_path, output_dir="extracted_pieces_edge"):
    """
    Alternative method using edge detection for better puzzle piece extraction.
    This method is better for pieces with clear edges against a uniform background.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    img_display = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Close gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} initial contours")
    
    # Filter and process contours
    min_area = 500
    max_area = img.shape[0] * img.shape[1] * 0.5
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Approximate contour to smooth out jagged edges
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            valid_contours.append(approx)
    
    print(f"Found {len(valid_contours)} valid puzzle pieces")
    
    # Extract and save pieces
    for idx, contour in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        piece = img[y_start:y_end, x_start:x_end]
        
        # Create mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        piece_mask = mask[y_start:y_end, x_start:x_end]
        
        # Create RGBA image with transparency
        piece_rgba = cv2.cvtColor(piece, cv2.COLOR_BGR2BGRA)
        piece_rgba[:, :, 3] = piece_mask
        
        piece_path = output_path / f"piece_{idx:03d}.png"
        cv2.imwrite(str(piece_path), piece_rgba)
        
        # Draw on visualization
        color = np.random.randint(0, 255, 3).tolist()
        cv2.drawContours(img_display, [contour], -1, color, 2)
        cv2.putText(img_display, str(idx), (x + w//2, y + h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save visualization and debug images
    cv2.imwrite(str(output_path / "visualization.jpg"), img_display)
    
    debug_path = output_path / "debug"
    debug_path.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_path / "edges.jpg"), edges)
    cv2.imwrite(str(debug_path / "closed.jpg"), closed)
    
    print(f"\nEdge-based extraction complete!")
    print(f"- Extracted {len(valid_contours)} pieces")
    print(f"- Pieces saved to: {output_path}")
    
    return len(valid_contours)


if __name__ == "__main__":
    # Path to the terracotta army puzzle image
    image_path = "terracotta_black_bg2/terracotta_army_pieces.png"
    
    # Output directory in the puzzle_solve folder
    output_base = "puzzle_solve/puzzle_pieces"
    
    print("Starting puzzle piece extraction...")
    print("=" * 50)
    
    # Method 1: Threshold-based extraction
    print("\nMethod 1: Threshold-based extraction")
    print("-" * 30)
    pieces_info = extract_puzzle_pieces(image_path, f"{output_base}/threshold_method")
    
    # Method 2: Edge-based extraction
    print("\n" + "=" * 50)
    print("Method 2: Edge-based extraction")
    print("-" * 30)
    num_pieces = extract_with_edge_detection(image_path, f"{output_base}/edge_method")
    
    print("\n" + "=" * 50)
    print("Extraction complete! Check both methods and use the one that works better.")
    print(f"Results saved in: {output_base}/")