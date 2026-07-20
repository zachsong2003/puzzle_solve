import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import os

import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict
import argparse
from PIL import Image

class PuzzlePiecePositionMapper:
    """
    Maps extracted puzzle pieces to their correct row/col positions by comparing
    with reference images (row*_col*.png).
    """
    
    def __init__(self, reference_dir: str, scrambled_image_path: str, model_path: str):
        # self.reference_dir = Path(reference_dir)
        self.scrambled_image_path = scrambled_image_path
        # self.reference_pieces = {}
        self.scrambled_img = cv2.imread(scrambled_image_path)
        
        if self.scrambled_img is None:
            raise ValueError(f"Could not load scrambled image from {scrambled_image_path}")
        
        # Load reference pieces
        # self._load_reference_pieces()

        self.model = YOLO(model_path)
        self.conf_threshold = 0.5
        
        # Class names for puzzle pieces
        self.class_pos = {
            # 0: 'row0_col0',
            # 1: 'row0_col1',
            # 2: 'row1_col0',
            # 3: 'row1_col1',
            # 4: 'row2_col0',
            # 5: 'row2_col1'
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
            4: (2, 0),
            5: (2, 1)
        }
        
        # Colors for visualization (BGR format for OpenCV)
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Yellow
        ]
        
        print(f"Model loaded from: {model_path}")
        print(f"Confidence threshold: {self.conf_threshold}")
        
    def _load_reference_pieces(self):
        """Load all reference pieces with row/col positions."""
        reference_files = list(self.reference_dir.glob("row*_col*.png"))
        
        if not reference_files:
            print(f"Warning: No reference pieces found in {self.reference_dir}")
            return
            
        print(f"Found {len(reference_files)} reference pieces")
        
        for file_path in reference_files:
            # Parse row and col from filename
            filename = file_path.stem
            parts = filename.split('_')
            
            try:
                row = int(parts[0].replace('row', ''))
                col = int(parts[1].replace('col', ''))
                
                img = cv2.imread(str(file_path))
                if img is not None:
                    self.reference_pieces[(row, col)] = {
                        'image': img,
                        'path': str(file_path),
                        'filename': file_path.name
                    }
                    print(f"Loaded reference piece: row {row}, col {col}")
            except (IndexError, ValueError) as e:
                print(f"Could not parse position from {filename}: {e}")
    
    def extract_pieces_from_scrambled(self, output_dir) -> List[Dict]:
        """Extract individual pieces from the scrambled image using edge detection."""
        gray = cv2.cvtColor(self.scrambled_img, cv2.COLOR_BGR2GRAY)
        
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
        
        print(f"Found {len(contours)} initial contours using edge detection")
        
        # Filter and extract pieces
        pieces = []
        min_area = 500  # Minimum area for a valid piece
        max_area = self.scrambled_img.shape[0] * self.scrambled_img.shape[1] * 0.5  # Maximum area (half of image)
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Approximate contour to smooth out jagged edges
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                valid_contours.append(approx)
        
        print(f"Filtered to {len(valid_contours)} valid contours")
        
        for idx, contour in enumerate(valid_contours):
            area = cv2.contourArea(contour)
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            padding = 10
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(self.scrambled_img.shape[1], x + w + padding)
            y_end = min(self.scrambled_img.shape[0], y + h + padding)
            
            # Extract piece
            piece_img = self.scrambled_img[y_start:y_end, x_start:x_end]
            
            # Create mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            piece_mask = mask[y_start:y_end, x_start:x_end]
            
            pieces.append({
                'index': idx,
                'image': piece_img,
                'mask': piece_mask,
                'contour': contour,
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'area': area,
                'area_by_bbox': w * h
            })
        
        print(f"Extracted {len(pieces)} pieces from scrambled image using edge detection")
        
        # Save debug images for inspection
        debug_dir = Path(f"{output_dir}/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "edges.jpg"), edges)
        cv2.imwrite(str(debug_dir / "dilated.jpg"), dilated)
        cv2.imwrite(str(debug_dir / "closed.jpg"), closed)
        print(f"Debug images saved to {debug_dir}")
        
        return pieces
    
    def match_piece_to_position(self, piece: Dict, reference: Dict) -> float:
        """
        Calculate similarity between an extracted piece and a reference piece.
        Returns a similarity score (higher is better).
        """
        piece_img = piece['image']
        ref_img = reference['image']
        
        # Resize reference to match piece size for comparison
        piece_h, piece_w = piece_img.shape[:2]
        ref_h, ref_w = ref_img.shape[:2]
        
        # Calculate scale factor
        scale = min(piece_w / ref_w, piece_h / ref_h)
        new_w = int(ref_w * scale)
        new_h = int(ref_h * scale)
        
        ref_resized = cv2.resize(ref_img, (new_w, new_h))
        
        # If sizes still don't match, pad or crop
        if piece_img.shape[:2] != ref_resized.shape[:2]:
            # Make them the same size
            min_h = min(piece_h, new_h)
            min_w = min(piece_w, new_w)
            piece_crop = piece_img[:min_h, :min_w]
            ref_crop = ref_resized[:min_h, :min_w]
        else:
            piece_crop = piece_img
            ref_crop = ref_resized
        
        # Method 1: Template matching
        piece_gray = cv2.cvtColor(piece_crop, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
        
        # Normalize images
        piece_norm = cv2.normalize(piece_gray, None, 0, 255, cv2.NORM_MINMAX)
        ref_norm = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Calculate correlation coefficient
        correlation = cv2.matchTemplate(piece_norm, ref_norm, cv2.TM_CCOEFF_NORMED)
        if correlation.size > 0:
            template_score = correlation[0, 0]
        else:
            template_score = 0
        
        # Method 2: Feature matching with ORB
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(piece_gray, None)
        kp2, des2 = orb.detectAndCompute(ref_gray, None)
        
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            feature_score = len(matches) / 100.0  # Normalize to 0-1 range
        else:
            feature_score = 0
        
        # Method 3: Histogram comparison
        hist1 = cv2.calcHist([piece_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        
        hist2 = cv2.calcHist([ref_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Combine scores (weighted average)
        combined_score = (template_score * 0.4 + feature_score * 0.3 + hist_score * 0.3)
        
        return combined_score
    
    def map_all_pieces(self, output_dir: str = "mapped_pieces"):
        """
        Map all extracted pieces to their row/col positions.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract pieces from scrambled image
        extracted_pieces = self.extract_pieces_from_scrambled(output_dir=output_dir)
        
        if not extracted_pieces:
            print("No pieces extracted from scrambled image")
            return
        
        # if not self.reference_pieces:
        #     print("No reference pieces loaded")
        #     return
        
        # Create mapping
        piece_mappings = []
        visualization = self.scrambled_img.copy()
        
        for piece in extracted_pieces:
            if piece['area'] < 15000:
                print(f"piece area < 15000: {piece}")
                continue
            best_match_pos = None
            best_score = -1

            piece_img = piece['image']
            results = self.model(piece_img, conf=self.conf_threshold)
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        # Get class and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if conf >= 0.5:
                            print(f"conf >= 0.5: {conf}")
                            best_score = conf
                            best_match_pos = self.class_pos[cls]                            
                            break
                        else:
                            print(f"conf < 0.5: {conf}") 
            
            # Compare with each reference piece
            # for (row, col), reference in self.reference_pieces.items():
                # score = self.match_piece_to_position(piece, reference)
                
                # if score > best_score:
                #     best_score = score
                #     best_match_pos = (row, col)
            
            if best_match_pos:
                row, col = best_match_pos
                piece['mapped_position'] = {'row': row, 'col': col}
                piece['match_score'] = best_score
                
                # Draw on visualization
                x, y, w, h = piece['bbox']
                color = (0, 255, 0) if best_score > 0.5 else (0, 165, 255)  # Green for good match, orange for uncertain
                cv2.rectangle(visualization, (x, y), (x+w, y+h), color, 2)
                
                # Add label
                label = f"R{row}C{col}"
                cv2.putText(visualization, label, (x+5, y+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add confidence score
                conf_text = f"{best_score:.2f}"
                cv2.putText(visualization, conf_text, (x+5, y+h-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Save individual piece with label
                piece_with_label = piece['image'].copy()
                piece_filename = f"piece_r{row}_c{col}_score{best_score:.2f}.png"
                cv2.imwrite(str(output_path / piece_filename), piece_with_label)

                cv2.putText(piece_with_label, f"Row {row}, Col {col}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                piece_label_filename = f"piece_r{row}_c{col}_label_score{best_score:.2f}.png"
                cv2.imwrite(str(output_path / piece_label_filename), piece_with_label)
                
                piece_mappings.append({
                    'piece_index': piece['index'],
                    'row': row,
                    'col': col,
                    'match_score': float(best_score),
                    'bbox': piece['bbox'],
                    'center': piece['center'],
                    'area': float(piece['area']),
                    'output_file': piece_filename
                })
                
                print(f"Piece {piece['index']}: Mapped to Row {row}, Col {col} (score: {best_score:.3f})")
        
        # Save visualization
        vis_path = output_path / "mapped_visualization.jpg"
        cv2.imwrite(str(vis_path), visualization)
        
        # Save mapping info as JSON
        mapping_info = {
            'total_pieces': len(extracted_pieces),
            'mapped_pieces': len(piece_mappings),
            # 'reference_pieces': len(self.reference_pieces),
            'mappings': piece_mappings
        }
        
        json_path = output_path / "piece_mappings.json"
        with open(json_path, 'w') as f:
            json.dump(mapping_info, f, indent=2)
        
        # Create summary image showing grid layout
        self._create_grid_summary(piece_mappings, extracted_pieces, output_path)
        
        print(f"\nMapping complete!")
        print(f"- Mapped {len(piece_mappings)} out of {len(extracted_pieces)} pieces")
        print(f"- Results saved to: {output_path}")
        print(f"- Visualization: {vis_path}")
        print(f"- Mapping data: {json_path}")
        
        return mapping_info
    
    def _create_grid_summary(self, mappings: List[Dict], pieces: List[Dict], output_path: Path):
        """Create a grid showing the reconstructed puzzle layout."""
        if not mappings:
            return
        
        # Find grid dimensions
        max_row = max(m['row'] for m in mappings)
        max_col = max(m['col'] for m in mappings)
        
        # Estimate piece size
        avg_width = np.mean([p['bbox'][2] for p in pieces])
        avg_height = np.mean([p['bbox'][3] for p in pieces])
        
        # Create grid image
        grid_width = int((max_col + 1) * avg_width)
        grid_height = int((max_row + 1) * avg_height)
        grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Place pieces in grid
        for mapping in mappings:
            row = mapping['row']
            col = mapping['col']
            
            # Find corresponding piece
            piece = next((p for p in pieces if p['index'] == mapping['piece_index']), None)
            if piece is None:
                continue
            
            # Calculate position in grid
            x_pos = int(col * avg_width)
            y_pos = int(row * avg_height)
            
            # Get piece image
            piece_img = piece['image']
            h, w = piece_img.shape[:2]
            
            # Resize if needed
            if w > avg_width or h > avg_height:
                scale = min(avg_width/w, avg_height/h) * 0.9
                new_w = int(w * scale)
                new_h = int(h * scale)
                piece_img = cv2.resize(piece_img, (new_w, new_h))
                h, w = new_h, new_w
            
            # Place in grid
            x_end = min(x_pos + w, grid_width)
            y_end = min(y_pos + h, grid_height)
            
            grid_img[y_pos:y_end, x_pos:x_end] = piece_img[:y_end-y_pos, :x_end-x_pos]
            
            # Draw grid lines
            cv2.rectangle(grid_img, (x_pos, y_pos), 
                         (int(x_pos + avg_width), int(y_pos + avg_height)), 
                         (200, 200, 200), 1)
            
            # Add label
            cv2.putText(grid_img, f"R{row}C{col}", (x_pos + 5, y_pos + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save grid summary
        grid_path = output_path / "reconstructed_grid.jpg"
        cv2.imwrite(str(grid_path), grid_img)
        print(f"- Grid reconstruction: {grid_path}")


def main():
    """Main function to run the piece position mapping."""
    # Paths
    reference_dir = "terracotta_black_bg2"
    scrambled_image = "terracotta_black_bg2/terracotta_army_pieces.png"
    output_dir = "mapped_pieces"
    
    print("=" * 60)
    print("PUZZLE PIECE POSITION MAPPING")
    print("=" * 60)
    print(f"Reference directory: {reference_dir}")
    print(f"Scrambled image: {scrambled_image}")
    print()
    
    try:
        # Create mapper and run mapping
        mapper = PuzzlePiecePositionMapper(reference_dir, scrambled_image)
        mapping_info = mapper.map_all_pieces(output_dir)
        
        if mapping_info:
            print("\n" + "=" * 60)
            print("MAPPING SUMMARY")
            print("=" * 60)
            print(f"Total pieces extracted: {mapping_info['total_pieces']}")
            print(f"Successfully mapped: {mapping_info['mapped_pieces']}")
            # print(f"Reference pieces available: {mapping_info['reference_pieces']}")
            
            # Show confidence statistics
            if mapping_info['mappings']:
                scores = [m['match_score'] for m in mapping_info['mappings']]
                print(f"\nMatch confidence scores:")
                print(f"  Average: {np.mean(scores):.3f}")
                print(f"  Min: {np.min(scores):.3f}")
                print(f"  Max: {np.max(scores):.3f}")
                
                high_conf = sum(1 for s in scores if s > 0.5)
                print(f"  High confidence matches (>0.5): {high_conf}/{len(scores)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()