# Camera Capture for Puzzle Solver

This document explains how to set up and use the camera capture functionality for the puzzle solver.

## Overview

The camera capture script ([`camera_capture.py`](camera_capture.py)) uses OpenCV to capture photos from your computer's camera and automatically sends them to the puzzle solving server for processing.

## Prerequisites

### 1. Install Dependencies

First, install all required dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python` - For camera capture and image processing
- `requests` - For HTTP communication with the server
- `pillow` - For image format conversion
- `numpy` - For image array operations
- `fastapi`, `uvicorn`, `openai`, `python-multipart` - For the server

### 2. Set Up OpenAI API Key

The puzzle solver uses OpenAI's image editing API. You need to set your API key:

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Camera Access

Ensure your computer's camera is:
- Connected and working
- Not being used by other applications
- Accessible to Python applications

## Usage

### Step 1: Start the Server

First, start the puzzle solving server:

```bash
uvicorn aiml.puzzle_solve.server:app --host 127.0.0.1 --port 8000 --reload
```

You should see output like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Step 2: Run the Camera Capture Script

In a new terminal, run the camera capture script:

```bash
python aiml/puzzle_solve/camera_capture.py
```

Or from the puzzle_solve directory:

```bash
cd aiml/puzzle_solve
python camera_capture.py
```

### Step 3: Use the Camera Interface

A camera preview window will open with the following controls:

| Key | Action |
|-----|--------|
| **SPACE** | Capture photo and send to puzzle solver |
| **S** | Save current frame locally (without sending) |
| **Q** or **ESC** | Quit the application |

## Workflow

1. **Camera Preview**: The script opens a live camera feed window
2. **Position Your Puzzle**: Point the camera at your jigsaw puzzle
3. **Capture**: Press SPACE to capture the current frame
4. **Processing**: The image is automatically sent to the server
5. **AI Processing**: OpenAI processes the puzzle (takes 10-30 seconds)
6. **Results**: Both original and solved images are saved locally

## Output Files

All captured and processed images are saved in the `captured_images/` directory:

- `captured_YYYYMMDD_HHMMSS.jpg` - Original captured images
- `solved_YYYYMMDD_HHMMSS.png` - AI-processed solved puzzles
- `manual_save_YYYYMMDD_HHMMSS.jpg` - Manually saved frames (S key)

## Configuration Options

### Custom Server URL

If your server is running on a different address:

```bash
python camera_capture.py http://localhost:8080/solve
```

### Camera Selection

By default, the script uses camera index 0 (primary camera). To use a different camera, modify the `camera_index` parameter in the script.

### Image Quality Settings

The script automatically sets optimal camera properties:
- Resolution: 1280x720
- Frame rate: 30 FPS
- Output format: PNG for server upload

## Troubleshooting

### Camera Issues

**Problem**: "Could not open camera"
- **Solution**: Ensure no other applications are using the camera
- **Solution**: Try different camera indices (0, 1, 2, etc.)
- **Solution**: Check camera permissions in system settings

**Problem**: Poor image quality
- **Solution**: Ensure good lighting conditions
- **Solution**: Clean camera lens
- **Solution**: Adjust camera position and distance

### Server Connection Issues

**Problem**: "Could not connect to server"
- **Solution**: Verify the server is running on http://127.0.0.1:8000
- **Solution**: Check firewall settings
- **Solution**: Ensure no other service is using port 8000

**Problem**: "Request timed out"
- **Solution**: OpenAI API can be slow (10-30 seconds is normal)
- **Solution**: Check internet connection
- **Solution**: Verify OpenAI API key is valid

### API Issues

**Problem**: "Server error: 500"
- **Solution**: Check OpenAI API key is set correctly
- **Solution**: Verify API key has sufficient credits
- **Solution**: Check server logs for detailed error messages

## Advanced Usage

### Batch Processing

To process multiple images quickly:
1. Use 'S' key to save multiple frames first
2. Then use SPACE to process them one by one
3. Or modify the script to process saved images

### Custom Image Sizes

The script defaults to 1024x1024 output. To change this, modify the `output_size` parameter in the `CameraPuzzleSolver` class:

```python
solver = CameraPuzzleSolver(output_size="512x512")
```

Available sizes: "512x512", "768x768", "1024x1024"

## Code Structure

The main components of [`camera_capture.py`](camera_capture.py):

- **`CameraPuzzleSolver`** - Main class handling all functionality
- **`initialize_camera()`** - Sets up camera connection
- **`capture_frame()`** - Captures individual frames
- **`send_to_server()`** - Handles HTTP communication
- **`save_solved_image()`** - Processes and saves results
- **`run()`** - Main application loop

## Integration with Existing System

The camera capture script integrates seamlessly with the existing puzzle solver:

- Uses the same `/solve` endpoint as the web interface
- Sends images in the same format (multipart/form-data)
- Receives responses in the same JSON format
- Compatible with all existing server features

## Performance Tips

1. **Good Lighting**: Ensure adequate lighting for clear captures
2. **Stable Position**: Use a tripod or stable surface for the camera
3. **Close Distance**: Position camera close enough to capture puzzle details
4. **Minimal Movement**: Keep puzzle and camera steady during capture
5. **Clean Background**: Use a contrasting background for better puzzle detection

## Security Notes

- Camera access requires appropriate system permissions
- Images are temporarily stored locally and sent to OpenAI
- Ensure your OpenAI API key is kept secure
- Local images are saved unencrypted in the `captured_images/` directory