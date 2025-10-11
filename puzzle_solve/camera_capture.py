#!/usr/bin/env python3
"""
Camera Capture Script for Puzzle Solver

This script uses OpenCV to capture photos from the computer's camera,
detects white paper region, crops to that region, and automatically 
sends them to the puzzle solving server at http://127.0.0.1:8000/solve

Usage:
    python camera_capture.py

Controls:
    - SPACE: Capture photo, detect paper, crop, and send to server
    - 'q' or ESC: Quit the application
    - 's': Save current frame locally (optional)
    - 'c': Toggle paper cropping on/off

Requirements:
    - Server must be running: uvicorn aiml.puzzle_solve.server:app --host 127.0.0.1 --port 8000
    - Camera must be accessible
"""

import cv2
import numpy as np
import requests
import base64
import json
import os
import time
from datetime import datetime
from io import BytesIO
from PIL import Image
import sys


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
    
    # Method 2: HSV-based detection for white/light colors
    # White has low saturation and high value
    # lower_white = np.array([0, 0, 180])  # Low saturation, high value
    # upper_white = np.array([180, 30, 255])  # Any hue, low saturation, high value
    # hsv_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # # Combine both masks
    # combined_mask = cv2.bitwise_or(white_mask, hsv_mask)

    combined_mask = white_mask
    
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


class CameraPuzzleSolver:
    def __init__(self, server_url="http://127.0.0.1:8000/solve", output_size="1024x1024"):
        """
        Initialize the camera puzzle solver.
        
        Args:
            server_url (str): URL of the puzzle solving server
            output_size (str): Output image size for the solver
        """
        self.server_url = server_url
        self.output_size = output_size
        self.camera = None
        self.window_name = "Puzzle Camera - SPACE: capture | C: toggle crop | Q: quit"
        self.crop_enabled = True  # Enable cropping by default
        
        # Create output directory for saved images
        self.output_dir = "captured_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def initialize_camera(self, camera_index=0):
        """
        Initialize the camera connection.
        
        Args:
            camera_index (int): Camera index (0 for default camera)
            
        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        try:
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                print(f"Error: Could not open camera {camera_index}")
                return False
                
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera {camera_index} initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def capture_frame(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray or None: Captured frame or None if failed
        """
        if not self.camera or not self.camera.isOpened():
            print("Error: Camera not initialized")
            return None
            
        ret, frame = self.camera.read()
        if not ret:
            print("Error: Failed to capture frame")
            return None
            
        return frame
    
    def save_image_locally(self, frame, prefix="captured"):
        """
        Save an image locally with timestamp.
        
        Args:
            frame (numpy.ndarray): Image frame to save
            prefix (str): Filename prefix
            
        Returns:
            str: Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Image saved locally: {filepath}")
        return filepath
    
    def process_with_paper_detection(self, frame):
        """
        Process frame with white paper detection and cropping.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (processed_frame, success_flag, message)
        """
        if not self.crop_enabled:
            return frame, True, "Cropping disabled"
        
        print("🔍 Detecting white paper...")
        paper_contour = detect_white_paper(frame)
        
        if paper_contour is not None:
            print("✅ White paper detected! Cropping to paper region...")
            try:
                cropped = crop_to_paper(frame, paper_contour)
                
                # Save both original and cropped for reference
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save original
                orig_path = os.path.join(self.output_dir, f"original_{timestamp}.jpg")
                cv2.imwrite(orig_path, frame)
                
                # Save cropped
                crop_path = os.path.join(self.output_dir, f"cropped_{timestamp}.jpg")
                cv2.imwrite(crop_path, cropped)
                
                print(f"📁 Original saved: {orig_path}")
                print(f"📁 Cropped saved: {crop_path}")
                
                return cropped, True, "Paper detected and cropped"
                
            except Exception as e:
                print(f"⚠️ Error cropping to paper: {e}")
                return frame, False, f"Crop error: {e}"
        else:
            print("⚠️ White paper not detected. Sending full image...")
            return frame, False, "Paper not detected"
    
    def frame_to_bytes(self, frame):
        """
        Convert OpenCV frame to bytes for HTTP upload.
        
        Args:
            frame (numpy.ndarray): OpenCV frame
            
        Returns:
            bytes: Image as bytes in PNG format
        """
        # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Convert to bytes
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return img_buffer.getvalue()
    
    def send_to_server(self, frame):
        """
        Send captured frame to the puzzle solving server.
        
        Args:
            frame (numpy.ndarray): Captured frame
            
        Returns:
            dict or None: Server response or None if failed
        """
        try:
            print("Preparing image for upload...")
            
            # Convert frame to bytes
            image_bytes = self.frame_to_bytes(frame)
            
            # Prepare the multipart form data
            files = {
                'image': ('puzzle.png', image_bytes, 'image/png')
            }
            data = {
                'size': self.output_size
            }
            
            print(f"Sending image to server: {self.server_url}")
            print("This may take 10-30 seconds...")
            
            # Send POST request
            response = requests.post(
                self.server_url,
                files=files,
                data=data,
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Image processed successfully!")
                return result
            else:
                print(f"❌ Server error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error details: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"Error response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out. Server may be busy.")
            return None
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to server. Make sure it's running at", self.server_url)
            return None
        except Exception as e:
            print(f"❌ Error sending to server: {e}")
            return None
    
    def save_solved_image(self, base64_data, prefix="solved"):
        """
        Save the solved image from base64 data.
        
        Args:
            base64_data (str): Base64 encoded image data
            prefix (str): Filename prefix
            
        Returns:
            str: Path to saved file
        """
        try:
            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
                
            print(f"✅ Solved image saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving solved image: {e}")
            return None
    
    def display_instructions(self):
        """Display usage instructions."""
        print("\n" + "="*60)
        print("🎯 PUZZLE CAMERA CAPTURE WITH PAPER DETECTION")
        print("="*60)
        print("📷 Camera preview will open in a new window")
        print("📄 White paper detection is ENABLED by default")
        print("⌨️  Controls:")
        print("   SPACE    - Capture photo, detect paper, crop, and send")
        print("   C        - Toggle paper cropping ON/OFF")
        print("   S        - Save current frame locally")
        print("   Q or ESC - Quit application")
        print("="*60)
        print("🔧 Make sure the server is running:")
        print("   uvicorn aiml.puzzle_solve.server:app --host 127.0.0.1 --port 8000")
        print("="*60)
    
    def run(self):
        """
        Main application loop.
        """
        self.display_instructions()
        
        # Initialize camera
        if not self.initialize_camera():
            print("❌ Failed to initialize camera. Exiting.")
            return
        
        print("\n🎥 Camera initialized. Opening preview window...")
        print(f"📄 Paper cropping: {'ENABLED' if self.crop_enabled else 'DISABLED'}")
        
        try:
            while True:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    print("❌ Failed to capture frame")
                    break
                
                # Create display frame with paper detection preview
                overlay_frame = frame.copy()
                
                # Try to detect paper in real-time for preview (if enabled)
                if self.crop_enabled:
                    paper_contour = detect_white_paper(frame)
                    if paper_contour is not None:
                        # Draw detected paper outline
                        cv2.drawContours(overlay_frame, [paper_contour], -1, (0, 255, 0), 2)
                        cv2.putText(overlay_frame, "Paper detected", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(overlay_frame, "Position white paper in view", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add text overlay with instructions
                status_text = f"Crop: {'ON' if self.crop_enabled else 'OFF'}"
                cv2.putText(overlay_frame, "SPACE: Capture | C: Toggle Crop | S: Save | Q: Quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay_frame, status_text, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if self.crop_enabled else (0, 165, 255), 2)
                
                # Display frame
                cv2.imshow(self.window_name, overlay_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Spacebar - capture and send
                    print("\n📸 Capturing image...")
                    
                    # Process with paper detection if enabled
                    processed_frame, crop_success, message = self.process_with_paper_detection(frame)
                    print(f"📄 {message}")
                    
                    # Save the processed frame
                    local_path = self.save_image_locally(processed_frame, "processed")
                    
                    # Send to server
                    result = self.send_to_server(processed_frame)
                    
                    if result and 'image_b64' in result:
                        # Save solved image
                        solved_path = self.save_solved_image(result['image_b64'], "solved")
                        print(f"🎉 Puzzle solved! Check: {solved_path}")
                    else:
                        print("❌ Failed to solve puzzle")
                    
                    break
                    
                    print("\n📷 Ready for next capture...")
                
                elif key == ord('c') or key == ord('C'):  # Toggle cropping
                    self.crop_enabled = not self.crop_enabled
                    status = "ENABLED" if self.crop_enabled else "DISABLED"
                    print(f"\n📄 Paper cropping {status}")
                
                elif key == ord('s') or key == ord('S'):  # Save current frame
                    self.save_image_locally(frame, "manual_save")
                
                elif key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC - quit
                    print("\n👋 Exiting...")
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Exiting...")
        
        finally:
            # Cleanup
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            print("🧹 Cleanup complete.")


def main():
    """Main entry point."""
    # Check if server URL is provided as argument
    server_url = "http://127.0.0.1:8000/solve"
    # server_url = "http://127.0.0.1:8000/solve-with-references"
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    
    # Create and run the camera solver
    solver = CameraPuzzleSolver(server_url=server_url)
    solver.run()


if __name__ == "__main__":
    main()