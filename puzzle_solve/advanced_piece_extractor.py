import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict
import json

class AdvancedPuzzlePieceExtractor:
    """
    Advanced puzzle piece extractor with multiple preprocessing and extraction methods.
    Specifically designed for puzzle pieces with convex and concave edges.
    """
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original_img.shape[:2]
        print(f"Loaded image: {self.width}x{self.height}")
        
    def preprocess_black_background(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image with black background to enhance piece detection.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Use Otsu's thresholding for better separation
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_pieces_watershed(self) -> List[np.ndarray]:
        """
        Use watershed algorithm for piece segmentation.
        Good for separating touching pieces.
        """
        img = self.original_img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(img, markers)
        
        # Extract contours from watershed boundaries
        contours = []
        for label in np.unique(markers):
            if label <= 1:  # Skip background and borders
                continue
            
            # Create mask for this label
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers == label] = 255
            
            # Find contour
            piece_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if piece_contours:
                contours.extend(piece_contours)
        
        return contours
    
    def detect_pieces_adaptive(self) -> List[np.ndarray]:
        """
        Use adaptive thresholding for better handling of varying lighting.
        """
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for edge-preserving smoothing
        smooth = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 3)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def detect_pieces_color_clustering(self) -> List[np.ndarray]:
        """
        Use color clustering (K-means) to separate pieces.
        Good when pieces have distinct colors.
        """
        img = self.original_img.copy()
        
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Reshape for k-means
        pixel_values = lab.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 10  # Number of clusters - adjust based on number of pieces
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reshape back
        labels = labels.reshape((img.shape[0], img.shape[1]))
        
        # Extract contours for each cluster
        contours = []
        for i in range(1, k):  # Skip background (label 0)
            mask = np.uint8(labels == i) * 255
            
            # Clean up mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            cluster_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cluster_contours)
        
        return contours
    
    def refine_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Refine contours to better match puzzle piece shapes.
        """
        refined = []
        
        # Calculate average area for filtering
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return []
        
        avg_area = np.median(areas)
        min_area = avg_area * 0.3
        max_area = avg_area * 3
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Approximate contour to smooth edges
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.005 * perimeter  # Small epsilon to preserve puzzle piece details
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if contour is reasonably convex (puzzle pieces aren't perfectly convex)
            hull = cv2.convexHull(approx)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Puzzle pieces typically have solidity between 0.7 and 0.95
            if 0.6 < solidity < 0.98:
                refined.append(approx)
        
        return refined
    
    def extract_pieces(self, method: str = "adaptive", output_dir: str = "extracted_pieces") -> Dict:
        """
        Extract puzzle pieces using specified method.
        
        Args:
            method: One of "adaptive", "watershed", "color", or "all"
            output_dir: Directory to save extracted pieces
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Select extraction method
        if method == "adaptive":
            contours = self.detect_pieces_adaptive()
        elif method == "watershed":
            contours = self.detect_pieces_watershed()
        elif method == "color":
            contours = self.detect_pieces_color_clustering()
        elif method == "all":
            # Try all methods and combine results
            contours = []
            contours.extend(self.detect_pieces_adaptive())
            contours.extend(self.detect_pieces_watershed())
            # Remove duplicates based on center points
            contours = self._remove_duplicate_contours(contours)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Found {len(contours)} initial contours using {method} method")
        
        # Refine contours
        contours = self.refine_contours(contours)
        print(f"Refined to {len(contours)} valid puzzle pieces")
        
        # Create visualization
        vis_img = self.original_img.copy()
        pieces_info = []
        
        # Extract each piece
        for idx, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            padding = 15
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(self.width, x + w + padding)
            y_end = min(self.height, y + h + padding)
            
            # Extract piece region
            piece = self.original_img[y_start:y_end, x_start:x_end]
            
            # Create mask for transparency
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            piece_mask = mask[y_start:y_end, x_start:x_end]
            
            # Apply smoothing to mask edges for better blending
            piece_mask = cv2.GaussianBlur(piece_mask, (5, 5), 0)
            
            # Create RGBA image
            piece_rgba = cv2.cvtColor(piece, cv2.COLOR_BGR2BGRA)
            piece_rgba[:, :, 3] = piece_mask
            
            # Save piece
            piece_path = output_path / f"piece_{idx:03d}.png"
            cv2.imwrite(str(piece_path), piece_rgba)
            
            # Save piece without transparency as well
            piece_jpg_path = output_path / f"piece_{idx:03d}_solid.jpg"
            cv2.imwrite(str(piece_jpg_path), piece)
            
            # Calculate piece properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Detect corners (potential connection points)
            corners = cv2.goodFeaturesToTrack(mask[y_start:y_end, x_start:x_end], 
                                             maxCorners=20, qualityLevel=0.01, 
                                             minDistance=10)
            num_corners = len(corners) if corners is not None else 0
            
            pieces_info.append({
                'index': idx,
                'filename': f"piece_{idx:03d}.png",
                'position': {'x': int(x), 'y': int(y)},
                'size': {'width': int(w), 'height': int(h)},
                'area': float(area),
                'perimeter': float(perimeter),
                'solidity': float(solidity),
                'num_corners': num_corners,
                'center': {'x': int(x + w//2), 'y': int(y + h//2)}
            })
            
            # Draw on visualization
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.drawContours(vis_img, [contour], -1, color, 2)
            cv2.putText(vis_img, str(idx), (x + w//2 - 10, y + h//2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save visualization
        vis_path = output_path / f"visualization_{method}.jpg"
        cv2.imwrite(str(vis_path), vis_img)
        
        # Save pieces information
        info_path = output_path / f"pieces_info_{method}.json"
        with open(info_path, 'w') as f:
            json.dump(pieces_info, f, indent=2)
        
        # Create summary
        summary = {
            'method': method,
            'total_pieces': len(contours),
            'output_directory': str(output_path),
            'visualization': str(vis_path),
            'pieces_info': str(info_path),
            'average_piece_area': float(np.mean([p['area'] for p in pieces_info])) if pieces_info else 0,
            'image_size': {'width': self.width, 'height': self.height}
        }
        
        return summary
    
    def _remove_duplicate_contours(self, contours: List[np.ndarray], threshold: float = 30) -> List[np.ndarray]:
        """
        Remove duplicate contours based on center distance.
        """
        if not contours:
            return []
        
        # Calculate centers
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
            else:
                centers.append(None)
        
        # Keep track of unique contours
        unique_indices = []
        for i, center_i in enumerate(centers):
            if center_i is None:
                continue
            
            is_unique = True
            for j in unique_indices:
                center_j = centers[j]
                if center_j is None:
                    continue
                
                # Calculate distance
                dist = np.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
                if dist < threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_indices.append(i)
        
        return [contours[i] for i in unique_indices]


def main():
    """
    Main function to run the advanced puzzle piece extraction.
    """
    image_path = "terracotta_black_bg2/terracotta_army_pieces.png"
    output_base = "puzzle_solve/puzzle_pieces"
    
    print("Advanced Puzzle Piece Extraction")
    print("=" * 50)
    
    try:
        extractor = AdvancedPuzzlePieceExtractor(image_path)
        
        # Try different methods
        methods = ["adaptive", "watershed", "color"]
        
        for method in methods:
            print(f"\nTrying {method} method...")
            print("-" * 30)
            
            output_dir = f"{output_base}/{method}_method"
            summary = extractor.extract_pieces(method=method, output_dir=output_dir)
            
            print(f"Extracted {summary['total_pieces']} pieces")
            print(f"Results saved to: {summary['output_directory']}")
            print(f"Average piece area: {summary['average_piece_area']:.2f} pixels²")
        
        print("\n" + "=" * 50)
        print("Extraction complete! Check the results in puzzle_solve/puzzle_pieces/")
        print("Each method folder contains:")
        print("  - Individual piece images (with transparency)")
        print("  - Solid piece images (without transparency)")
        print("  - Visualization showing all detected pieces")
        print("  - JSON file with detailed piece information")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()