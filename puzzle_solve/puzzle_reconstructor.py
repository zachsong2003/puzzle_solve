import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import os

class PuzzleReconstructor:
    """
    Reconstructs puzzle by connecting pieces based on their convex/concave edges.
    """
    
    def __init__(self, mapped_pieces_dir: str = "puzzle_solve/mapped_pieces"):
        self.mapped_pieces_dir = Path(mapped_pieces_dir)
        self.pieces_data = {}
        self.piece_images = {}
        self.edge_profiles = {}
        
        # Load mapping data
        self._load_mapping_data()
        
    def _load_mapping_data(self):
        """Load the piece mapping data and images."""
        json_path = self.mapped_pieces_dir / "piece_mappings.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            self.mapping_info = json.load(f)
        
        print(f"Loaded mapping info for {len(self.mapping_info['mappings'])} pieces")
        
        # Load individual piece images
        for mapping in self.mapping_info['mappings']:
            if mapping['area'] < 10000:
                print(f"piece area < 10000: {mapping}")
                continue

            piece_file = self.mapped_pieces_dir / mapping['output_file']
            if piece_file.exists():
                img = cv2.imread(str(piece_file))
                if img is not None:
                    row = mapping['row']
                    col = mapping['col']
                    self.piece_images[(row, col)] = img
                    self.pieces_data[(row, col)] = mapping
    
    def detect_edge_profile(self, piece_img: np.ndarray, edge_side: str) -> np.ndarray:
        """
        Detect the edge profile of a piece for a specific side.
        
        Args:
            piece_img: The piece image
            edge_side: One of 'top', 'bottom', 'left', 'right'
        
        Returns:
            Edge profile as a 1D array
        """
        gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY) if len(piece_img.shape) == 3 else piece_img
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Extract edge profile based on side
        h, w = edges.shape
        
        if edge_side == 'top':
            # Find the topmost edge pixels for each column
            profile = []
            for x in range(w):
                col = edges[:h//3, x]  # Look only in top third
                edge_pixels = np.where(col > 0)[0]
                if len(edge_pixels) > 0:
                    profile.append(edge_pixels[0])
                else:
                    profile.append(0)
                    
        elif edge_side == 'bottom':
            # Find the bottommost edge pixels for each column
            profile = []
            for x in range(w):
                col = edges[2*h//3:, x]  # Look only in bottom third
                edge_pixels = np.where(col > 0)[0]
                if len(edge_pixels) > 0:
                    profile.append(2*h//3 + edge_pixels[-1])
                else:
                    profile.append(h-1)
                    
        elif edge_side == 'left':
            # Find the leftmost edge pixels for each row
            profile = []
            for y in range(h):
                row = edges[y, :w//3]  # Look only in left third
                edge_pixels = np.where(row > 0)[0]
                if len(edge_pixels) > 0:
                    profile.append(edge_pixels[0])
                else:
                    profile.append(0)
                    
        elif edge_side == 'right':
            # Find the rightmost edge pixels for each row
            profile = []
            for y in range(h):
                row = edges[y, 2*w//3:]  # Look only in right third
                edge_pixels = np.where(row > 0)[0]
                if len(edge_pixels) > 0:
                    profile.append(2*w//3 + edge_pixels[-1])
                else:
                    profile.append(w-1)
        else:
            profile = []
        
        return np.array(profile)
    
    def match_edges(self, edge1: np.ndarray, edge2: np.ndarray) -> float:
        """
        Calculate how well two edges match (one convex, one concave).
        
        Returns:
            Match score (0-1, higher is better)
        """
        if len(edge1) == 0 or len(edge2) == 0:
            return 0
        
        # Ensure arrays are valid
        edge1 = np.array(edge1, dtype=np.float32)
        edge2 = np.array(edge2, dtype=np.float32)
        
        # Resize to same length if needed using interpolation
        if len(edge1) != len(edge2):
            target_len = min(len(edge1), len(edge2))
            if target_len < 2:
                return 0
            
            # Use numpy interpolation instead of cv2.resize for 1D arrays
            x1 = np.linspace(0, len(edge1)-1, len(edge1))
            x1_new = np.linspace(0, len(edge1)-1, target_len)
            edge1 = np.interp(x1_new, x1, edge1)
            
            x2 = np.linspace(0, len(edge2)-1, len(edge2))
            x2_new = np.linspace(0, len(edge2)-1, target_len)
            edge2 = np.interp(x2_new, x2, edge2)
        
        # Invert one edge (to match convex with concave)
        if len(edge2) > 0:
            edge2_inverted = np.max(edge2) - edge2 + np.min(edge2)
        else:
            return 0
        
        # Calculate correlation
        if np.std(edge1) > 0 and np.std(edge2_inverted) > 0:
            correlation = np.corrcoef(edge1, edge2_inverted)[0, 1]
            score = max(0, correlation)  # Keep only positive correlations
        else:
            score = 0
        
        return score
    
    def find_piece_transformations(self) -> Dict[Tuple[int, int], Dict]:
        """
        Calculate optimal transformations for each piece to connect with neighbors.
        """
        transformations = {}
        
        for (row, col), piece_img in self.piece_images.items():
            transform = {
                'offset_x': 0,
                'offset_y': 0,
                'rotation': 0
            }
            
            # Check right neighbor
            if (row, col + 1) in self.piece_images:
                right_neighbor = self.piece_images[(row, col + 1)]
                
                # Get edge profiles
                my_right = self.detect_edge_profile(piece_img, 'right')
                neighbor_left = self.detect_edge_profile(right_neighbor, 'left')
                
                # Calculate optimal horizontal offset
                if len(my_right) > 0 and len(neighbor_left) > 0:
                    # Find offset that maximizes edge matching
                    best_offset = 0
                    best_score = 0
                    
                    for offset in range(-10, 10):
                        if offset < 0:
                            edge1 = my_right[-offset:]
                            edge2 = neighbor_left[:len(neighbor_left)+offset]
                        else:
                            edge1 = my_right[:len(my_right)-offset]
                            edge2 = neighbor_left[offset:]
                        
                        if len(edge1) > 0 and len(edge2) > 0:
                            score = self.match_edges(edge1, edge2)
                            if score > best_score:
                                best_score = score
                                best_offset = offset
                    
                    transform['offset_x'] = best_offset
            
            # Check bottom neighbor
            if (row + 1, col) in self.piece_images:
                bottom_neighbor = self.piece_images[(row + 1, col)]
                
                # Get edge profiles
                my_bottom = self.detect_edge_profile(piece_img, 'bottom')
                neighbor_top = self.detect_edge_profile(bottom_neighbor, 'top')
                
                # Calculate optimal vertical offset
                if len(my_bottom) > 0 and len(neighbor_top) > 0:
                    best_offset = 0
                    best_score = 0
                    
                    for offset in range(-10, 10):
                        if offset < 0:
                            edge1 = my_bottom[-offset:]
                            edge2 = neighbor_top[:len(neighbor_top)+offset]
                        else:
                            edge1 = my_bottom[:len(my_bottom)-offset]
                            edge2 = neighbor_top[offset:]
                        
                        if len(edge1) > 0 and len(edge2) > 0:
                            score = self.match_edges(edge1, edge2)
                            if score > best_score:
                                best_score = score
                                best_offset = offset
                    
                    transform['offset_y'] = best_offset
            
            transformations[(row, col)] = transform
        
        return transformations
    
    def reconstruct_with_edge_matching(self, output_path: str = None):
        """
        Reconstruct the puzzle with pieces connected by their edges.
        """
        if not self.piece_images:
            print("No piece images loaded")
            return None
        
        # Find grid dimensions
        max_row = max(pos[0] for pos in self.piece_images.keys())
        max_col = max(pos[1] for pos in self.piece_images.keys())
        
        print(f"Grid dimensions: {max_row + 1} rows x {max_col + 1} columns")
        
        # Get typical piece dimensions
        piece_heights = []
        piece_widths = []
        for img in self.piece_images.values():
            h, w = img.shape[:2]
            piece_heights.append(h)
            piece_widths.append(w)
        
        avg_height = int(np.median(piece_heights))
        avg_width = int(np.median(piece_widths))
        
        # Calculate transformations for edge matching
        print("Calculating edge-based transformations...")
        transformations = self.find_piece_transformations()
        
        # Create canvas with some extra padding for adjustments
        padding = 50
        canvas_height = (max_row + 1) * avg_height + 2 * padding
        canvas_width = (max_col + 1) * avg_width + 2 * padding
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Place pieces with edge-based adjustments
        piece_positions = {}
        overlap_compensation = 0.9  # Slight overlap to connect pieces better
        
        for row in range(max_row + 1):
            for col in range(max_col + 1):
                if (row, col) not in self.piece_images:
                    continue
                
                piece_img = self.piece_images[(row, col)]
                h, w = piece_img.shape[:2]
                
                # Calculate base position
                base_x = padding + int(col * avg_width * overlap_compensation)
                base_y = padding + int(row * avg_height * overlap_compensation)
                
                # Apply transformations
                if (row, col) in transformations:
                    trans = transformations[(row, col)]
                    base_x += trans['offset_x']
                    base_y += trans['offset_y']
                
                # Apply fine-tuning based on previous pieces
                if col > 0 and (row, col-1) in piece_positions:
                    # Align with left neighbor
                    prev_x, prev_y, prev_w, prev_h = piece_positions[(row, col-1)]
                    base_x = prev_x + int(prev_w * overlap_compensation)
                
                if row > 0 and (row-1, col) in piece_positions:
                    # Align with top neighbor
                    prev_x, prev_y, prev_w, prev_h = piece_positions[(row-1, col)]
                    base_y = prev_y + int(prev_h * overlap_compensation)
                
                # Store position
                piece_positions[(row, col)] = (base_x, base_y, w, h)
                
                # Blend piece into canvas
                x_end = min(base_x + w, canvas_width)
                y_end = min(base_y + h, canvas_height)
                
                # Extract the region where the piece will be placed
                region = canvas[base_y:y_end, base_x:x_end]
                piece_region = piece_img[:y_end-base_y, :x_end-base_x]
                
                # Create mask for non-white pixels (actual piece content)
                gray = cv2.cvtColor(piece_region, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                
                # Dilate mask slightly to ensure good coverage
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                # Apply piece using mask
                mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                canvas[base_y:y_end, base_x:x_end] = (
                    region * (1 - mask_3channel) + piece_region * mask_3channel
                ).astype(np.uint8)
        
        # Crop to remove excess padding
        # Find the actual content boundaries
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_canvas, 235, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the bounding box of all content
            x_min = canvas_width
            y_min = canvas_height
            x_max = 0
            y_max = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Crop with small margin
            margin = 10
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(canvas_width, x_max + margin)
            y_max = min(canvas_height, y_max + margin)
            
            canvas = canvas[y_min:y_max, x_min:x_max]
        
        # Apply final smoothing to blend edges
        canvas = cv2.bilateralFilter(canvas, 5, 50, 50)
        
        # Save the result
        if output_path is None:
            output_path = str(self.mapped_pieces_dir / "reconstructed_grid_connected.jpg")
        
        cv2.imwrite(output_path, canvas)
        print(f"Reconstructed puzzle saved to: {output_path}")
        
        # Also create a version with visible grid lines for reference
        grid_canvas = canvas.copy()
        for row in range(max_row + 1):
            for col in range(max_col + 1):
                if (row, col) in piece_positions:
                    x, y, w, h = piece_positions[(row, col)]
                    # Adjust for cropping
                    x -= x_min if 'x_min' in locals() else 0
                    y -= y_min if 'y_min' in locals() else 0
                    
                    # Draw subtle grid lines
                    cv2.rectangle(grid_canvas, (x, y), (x+w, y+h), (200, 200, 200), 1)
                    
                    # Add position label
                    label = f"R{row}C{col}"
                    cv2.putText(grid_canvas, label, (x+5, y+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        grid_output_path = str(self.mapped_pieces_dir / "reconstructed_grid_with_labels.jpg")
        cv2.imwrite(grid_output_path, grid_canvas)
        print(f"Grid with labels saved to: {grid_output_path}")
        
        return canvas
    
    def create_seamless_reconstruction(self):
        """
        Create a more seamless reconstruction using advanced blending techniques.
        """
        if not self.piece_images:
            print("No piece images loaded")
            return None
        
        # Find grid dimensions
        max_row = max(pos[0] for pos in self.piece_images.keys())
        max_col = max(pos[1] for pos in self.piece_images.keys())
        
        # Get reference piece dimensions
        piece_heights = [img.shape[0] for img in self.piece_images.values()]
        piece_widths = [img.shape[1] for img in self.piece_images.values()]
        
        target_height = int(np.median(piece_heights))
        target_width = int(np.median(piece_widths))
        
        # Create the final canvas
        final_height = (max_row + 1) * target_height
        final_width = (max_col + 1) * target_width
        canvas = np.zeros((final_height, final_width, 3), dtype=np.float32)
        weight_map = np.zeros((final_height, final_width), dtype=np.float32)
        
        print("Creating seamless reconstruction with blending...")
        
        for row in range(max_row + 1):
            for col in range(max_col + 1):
                if (row, col) not in self.piece_images:
                    continue
                
                piece_img = self.piece_images[(row, col)].astype(np.float32)
                
                # Resize piece to target dimensions
                piece_resized = cv2.resize(piece_img, (target_width, target_height))
                
                # Calculate position
                y_start = row * target_height
                x_start = col * target_width
                
                # Create weight mask with smooth edges (feathering)
                h, w = target_height, target_width
                weight = np.ones((h, w), dtype=np.float32)
                
                # Create gradient weights for edges (for smooth blending)
                fade_width = 20  # Width of the fade region
                
                # Top edge
                for i in range(min(fade_width, h//4)):
                    weight[i, :] *= (i / fade_width)
                
                # Bottom edge
                for i in range(min(fade_width, h//4)):
                    weight[-(i+1), :] *= (i / fade_width)
                
                # Left edge
                for i in range(min(fade_width, w//4)):
                    weight[:, i] *= (i / fade_width)
                
                # Right edge
                for i in range(min(fade_width, w//4)):
                    weight[:, -(i+1)] *= (i / fade_width)
                
                # Add piece to canvas with weighted blending
                for c in range(3):
                    canvas[y_start:y_start+h, x_start:x_start+w, c] += piece_resized[:, :, c] * weight
                
                weight_map[y_start:y_start+h, x_start:x_start+w] += weight
        
        # Normalize by weights
        for c in range(3):
            canvas[:, :, c] /= np.maximum(weight_map, 1.0)
        
        # Convert back to uint8
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        
        # Apply slight Gaussian blur at the seams
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)
        
        # Save the seamless result
        seamless_path = str(self.mapped_pieces_dir / "reconstructed_seamless.jpg")
        cv2.imwrite(seamless_path, canvas)
        print(f"Seamless reconstruction saved to: {seamless_path}")
        
        return canvas


def main():
    """Main function to run the puzzle reconstruction."""
    
    print("=" * 60)
    print("PUZZLE RECONSTRUCTION WITH EDGE MATCHING")
    print("=" * 60)
    
    try:
        # Create reconstructor
        reconstructor = PuzzleReconstructor("puzzle_solve/mapped_pieces")
        
        # Create edge-matched reconstruction
        print("\n1. Creating edge-matched reconstruction...")
        reconstructor.reconstruct_with_edge_matching()
        
        # Create seamless reconstruction
        print("\n2. Creating seamless blended reconstruction...")
        reconstructor.create_seamless_reconstruction()
        
        print("\n" + "=" * 60)
        print("RECONSTRUCTION COMPLETE!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - reconstructed_grid_connected.jpg: Edge-matched reconstruction")
        print("  - reconstructed_grid_with_labels.jpg: With position labels")
        print("  - reconstructed_seamless.jpg: Seamless blended version")
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()