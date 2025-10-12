# Puzzle Solver

AI-powered jigsaw puzzle solver using OpenAI's image editing API.

## Features

- **Web Interface**: Upload puzzle images via drag-and-drop, paste, or file selection
- **Camera Capture**: Use your computer's camera to capture puzzle photos directly
- **AI Processing**: Leverages OpenAI's GPT-4 Vision to reconstruct solved puzzles
- **Multiple Formats**: Supports various image formats and output sizes

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Start the Server

```bash
uvicorn puzzle_solve.server:app --host 127.0.0.1 --port 8000 --reload
```

## Usage Options

### Option 1: Web Interface

1. Open your browser to `<path_to_puzzle_solve>/index.html`
2. Upload or paste a puzzle image
3. Click "Solve" and wait for the AI to process
4. Download the solved puzzle

### Option 2: Camera Capture

1. Run the camera capture script:
   ```bash
   python puzzle_solve/camera_capture.py
   ```
2. Position your puzzle in front of the camera
3. Wait for the green box for cropping ready, press SPACE to capture and automatically send to the solver
4. Results are saved in the `captured_images/` directory

For detailed camera setup instructions, see [`CAMERA_SETUP.md`](CAMERA_SETUP.md).

## Testing

Run the test suite to verify everything is working:

```bash
python test_camera.py
```

## Files

- [`server.py`](server.py) - FastAPI server with `/solve` endpoint
- [`index.html`](index.html) - Web interface for manual uploads
- [`camera_capture.py`](camera_capture.py) - OpenCV camera capture script
- [`requirements.txt`](requirements.txt) - Python dependencies
- [`CAMERA_SETUP.md`](CAMERA_SETUP.md) - Detailed camera setup guide
- [`test_camera.py`](test_camera.py) - Test script for camera functionality

## API

### POST /solve

Upload an image to be processed by the puzzle solver.

**Parameters:**

- `image` (file): Image file containing the puzzle
- `size` (string, optional): Output size ("512x512", "768x768", "1024x1024")

**Response:**

```json
{
  "image_b64": "base64-encoded-solved-image"
}
```

## Requirements

- Python 3.7+
- OpenAI API key with image editing access
- Camera access (for camera capture feature)
- Internet connection for OpenAI API calls

## Troubleshooting

- **Camera issues**: Check camera permissions and ensure no other apps are using it
- **Server errors**: Verify OpenAI API key is set and valid
- **Slow processing**: OpenAI API typically takes 10-30 seconds per image

## reconstruct image by opencv

```
python puzzle_solve/complete_puzzle_solver.py --output_dir mapped_pieces_camera
```

## test piece mapping by opencv

```
python puzzle_solve/run_extraction.py
python puzzle_solve/piece_position_mapper.py
python puzzle_solve/puzzle_reconstructor.py
```

