#!/usr/bin/env python3
"""
Standalone test application for white paper detection.
Tests different detection methods and shows results.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def detect_white_paper(image):
    """
    Detect white paper rectangle in the image using color-based detection.
    White paper on grey background.
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Also work with grayscale for brightness detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Method 1: Direct brightness thresholding for white paper
    # White paper should be brighter than grey background
    _, white_mask = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
    
    # Method 2: HSV-based detection for white/light colors
    # White has low saturation and high value
    lower_white = np.array([0, 0, 180])  # Low saturation, high value
    upper_white = np.array([180, 30, 255])  # Any hue, low saturation, high value
    hsv_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine both masks
    combined_mask = cv2.bitwise_or(white_mask, hsv_mask)
    
    # Clean up the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    
    # Remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilate to ensure edges are connected
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, combined_mask
    
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
            
            return approx, combined_mask
    
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
            return box, combined_mask
    
    return None, combined_mask

def detect_white_paper_advanced(image):
    """
    Alternative detection method using LAB color space.
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
        return None, binary
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Check if it's large enough
    if area > (image.shape[0] * image.shape[1] * 0.1):
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box).reshape(-1, 1, 2)
        return box, binary
    
    return None, binary

def test_with_different_thresholds(image):
    """
    Test with different brightness thresholds to find optimal value.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    results = []
    thresholds = [140, 150, 160, 170, 180, 190]
    
    for thresh_val in thresholds:
        _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            # Get bounding rectangle
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int32(box).reshape(-1, 1, 2)
            
            results.append({
                'threshold': thresh_val,
                'area': area,
                'contour': box,
                'mask': mask
            })
    
    return results

def crop_to_paper(image, paper_contour):
    """
    Crop and perspective-correct the image to just the paper region.
    """
    # Reshape contour points
    pts = paper_contour.reshape(4, 2)
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and diff to find corners
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    # Compute width and height
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply transform
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def main():
    """
    Main test function.
    """
    print("=" * 60)
    print("WHITE PAPER DETECTION TEST")
    print("=" * 60)
    
    # Default test image
    image_path = "terracotta_black_bg2/terracotta_army_pieces_bg.png"
    
    # Allow command line argument for different image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        print("\nUsage: python test_detect_white_paper.py [image_path]")
        return
    
    print(f"Testing image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Resize for display if too large
    display_scale = 1.0
    if image.shape[1] > 1200:
        display_scale = 1200.0 / image.shape[1]
        display_image = cv2.resize(image, None, fx=display_scale, fy=display_scale)
    else:
        display_image = image.copy()
    
    print("\n" + "-" * 40)
    print("METHOD 1: Color-based detection (HSV + Brightness)")
    print("-" * 40)
    
    # Test Method 1
    paper_contour, mask1 = detect_white_paper(image)
    
    if paper_contour is not None:
        print("✓ Paper detected successfully!")
        print(f"  Contour shape: {paper_contour.shape}")
        print(f"  Number of corners: {len(paper_contour)}")
        
        # Draw on display image
        result1 = display_image.copy()
        scaled_contour = (paper_contour * display_scale).astype(np.int32)
        cv2.drawContours(result1, [scaled_contour], -1, (0, 255, 0), 3)
        cv2.putText(result1, "Method 1: Color Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show mask
        mask1_resized = cv2.resize(mask1, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Method 1 - Mask", mask1_resized)
        cv2.imshow("Method 1 - Result", result1)
    else:
        print("✗ Paper not detected")
    
    print("\n" + "-" * 40)
    print("METHOD 2: LAB color space detection")
    print("-" * 40)
    
    # Test Method 2
    paper_contour2, mask2 = detect_white_paper_advanced(image)
    
    if paper_contour2 is not None:
        print("✓ Paper detected successfully!")
        print(f"  Contour shape: {paper_contour2.shape}")
        
        # Draw on display image
        result2 = display_image.copy()
        scaled_contour2 = (paper_contour2 * display_scale).astype(np.int32)
        cv2.drawContours(result2, [scaled_contour2], -1, (255, 0, 0), 3)
        cv2.putText(result2, "Method 2: LAB Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show mask
        mask2_resized = cv2.resize(mask2, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Method 2 - Mask", mask2_resized)
        cv2.imshow("Method 2 - Result", result2)
    else:
        print("✗ Paper not detected")
    
    print("\n" + "-" * 40)
    print("METHOD 3: Testing different thresholds")
    print("-" * 40)
    
    # Test different thresholds
    threshold_results = test_with_different_thresholds(image)
    
    for i, result in enumerate(threshold_results):
        print(f"Threshold {result['threshold']}: Area = {result['area']:.0f} pixels")
        
        # Show result
        result_img = display_image.copy()
        scaled_contour = (result['contour'] * display_scale).astype(np.int32)
        cv2.drawContours(result_img, [scaled_contour], -1, (0, 165, 255), 2)
        cv2.putText(result_img, f"Threshold: {result['threshold']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        window_name = f"Threshold {result['threshold']}"
        cv2.imshow(window_name, result_img)
    
    # If paper was detected, show cropped result
    if paper_contour is not None:
        print("\n" + "-" * 40)
        print("CROPPING TO PAPER REGION")
        print("-" * 40)
        
        cropped = crop_to_paper(image, paper_contour)
        print(f"Cropped image size: {cropped.shape[1]}x{cropped.shape[0]}")
        
        # Resize for display
        if cropped.shape[1] > 800:
            crop_scale = 800.0 / cropped.shape[1]
            cropped_display = cv2.resize(cropped, None, fx=crop_scale, fy=crop_scale)
        else:
            cropped_display = cropped
        
        cv2.imshow("Cropped Result", cropped_display)
        
        # Save cropped result
        output_path = "puzzle_solve/captured_images/test_cropped_result.jpg"
        cv2.imwrite(output_path, cropped)
        print(f"Cropped image saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nPress any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()