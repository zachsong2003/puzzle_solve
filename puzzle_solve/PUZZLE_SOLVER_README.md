# Puzzle Piece Extraction and Reconstruction System

This system provides a complete pipeline for extracting individual puzzle pieces from a scrambled image and reconstructing the puzzle by matching pieces to their correct positions.

## Features

### 1. **Piece Extraction** (`extract_puzzle_pieces.py`)
- Multiple extraction methods:
  - Threshold-based extraction
  - Edge detection-based extraction (Canny edge detection)
- Handles puzzle pieces with convex and concave edges
- Saves individual pieces with transparency

### 2. **Advanced Extraction** (`advanced_piece_extractor.py`)
- Advanced methods:
  - Adaptive thresholding
  - Watershed segmentation
  - Color clustering
- Better handling of varying lighting conditions
- Piece property analysis (area, perimeter, solidity)

### 3. **Position Mapping** (`piece_position_mapper.py`)
- Maps extracted pieces to their correct row/column positions
- Uses multiple matching techniques:
  - Template matching
  - Feature matching (ORB)
  - Histogram comparison
- Generates confidence scores for each mapping
- Creates visualization with position labels

### 4. **Puzzle Reconstruction** (`puzzle_reconstructor.py`)
- **Edge-Matched Reconstruction**: Connects pieces by matching convex/concave edges
- **Seamless Reconstruction**: Uses weighted blending for smooth transitions
- Edge profile detection and matching
- Automatic alignment and positioning

### 5. **Complete Pipeline** (`complete_puzzle_solver.py`)
- Runs the entire process automatically:
  1. Extracts pieces using edge detection
  2. Maps pieces to row/column positions
  3. Reconstructs puzzle with edge matching
  4. Creates seamless blended version

## Usage

### Quick Start - Complete Pipeline

```bash
cd puzzle_solve
python complete_puzzle_solver.py
```

This will:
1. Extract pieces from the scrambled terracotta army image
2. Match each piece to its row/column position
3. Create multiple reconstruction versions

### Individual Components

#### Extract Pieces Only
```bash
python extract_puzzle_pieces.py
```

#### Advanced Extraction
```bash
python advanced_piece_extractor.py
```

#### Map Pieces to Positions
```bash
python piece_position_mapper.py
```

#### Reconstruct from Mapped Pieces
```bash
python puzzle_reconstructor.py
```

## Input Requirements

1. **Scrambled Image**: `terracotta_army_pieces.png`
   - Contains all puzzle pieces in scrambled positions
   - Should have clear edges between pieces

2. **Reference Pieces** (optional): `row{n}_col{m}.png`
   - Individual piece images in correct positions
   - Used for position mapping
   - Named with row and column indices

## Output Files

The system generates several output files in `puzzle_solve/mapped_pieces/`:

### Individual Pieces
- `piece_r{row}_c{col}_score{confidence}.png` - Individual pieces with position labels

### Visualizations
- `mapped_visualization.jpg` - Shows all detected pieces with position labels
- `reconstructed_grid.jpg` - Basic grid reconstruction
- `reconstructed_grid_connected.jpg` - Edge-matched reconstruction
- `reconstructed_grid_with_labels.jpg` - Reconstruction with position labels
- `reconstructed_seamless.jpg` - Seamless blended reconstruction

### Data Files
- `piece_mappings.json` - Detailed mapping information including:
  - Piece positions
  - Confidence scores
  - Bounding boxes
  - Area measurements

### Debug Files
- `debug/edges.jpg` - Edge detection result
- `debug/dilated.jpg` - Dilated edges
- `debug/closed.jpg` - Morphologically closed edges

## Key Algorithms

### Edge Detection Method
- Uses Canny edge detection for finding piece boundaries
- Morphological operations to connect broken edges
- Contour approximation for smoothing

### Edge Profile Matching
- Extracts edge profiles from each side of pieces
- Matches convex edges with concave edges
- Calculates correlation scores for alignment

### Seamless Blending
- Uses weighted masks with feathered edges
- Gaussian blur for smooth transitions
- Overlap compensation for better connections

## Configuration

Key parameters can be adjusted in the code:

- **Edge Detection**:
  - Canny thresholds: `(30, 100)`
  - Dilation iterations: `2`
  - Minimum piece area: `500` pixels

- **Reconstruction**:
  - Overlap compensation: `0.9` (10% overlap)
  - Fade width for blending: `20` pixels
  - Edge matching offset range: `[-10, 10]` pixels

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- Pillow

Install dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### No pieces detected
- Check if the image has sufficient contrast
- Adjust Canny edge detection thresholds
- Try different extraction methods

### Poor position mapping
- Ensure reference pieces are correctly named
- Check if pieces have sufficient distinctive features
- Adjust matching score thresholds

### Gaps in reconstruction
- Increase overlap compensation factor
- Adjust edge matching parameters
- Try seamless blending method

## Example Results

The system successfully:
1. Extracts individual puzzle pieces from scrambled images
2. Maps each piece to its correct position using reference images
3. Reconstructs the puzzle with connected edges
4. Creates seamless blended versions

Best results are achieved when:
- Pieces have clear edges
- Reference images are available
- Lighting is consistent across the image