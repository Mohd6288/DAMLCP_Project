from py5canvas import *
from video_processor import VideoProcessor
import cv2

# Parameters for GUI
def parameters():
    return {
        "Threshold1": (100, {"min": 0, "max": 255}),  # Canny edge threshold 1
        "Threshold2": (200, {"min": 0, "max": 255}),  # Canny edge threshold 2
        "Smoothing": (1.0, {"min": 0.1, "max": 5.0}),  # Smoothing value
        "Face Detection": False,  # Enable/disable face detection
        "Hand Tracking": True,  # Enable/disable hand tracking
        "Video Border Color": ([255, 255, 255], {"type": "color"}),  # Color for video border
        "Take Photo": False,  # Button for capturing a photo
    }

# Global Video Processor
video_processor = None

def setup():
    global video_processor
    create_canvas(512, 512)
    video_processor = VideoProcessor(params)  # Initialize with parameters

def draw():
    global video_processor
    background(params.video_border_color)  # Set border color

    # Process the video feed
    frame = video_processor.process_frame()
    if frame is not None:
        frame_scaled = cv2.resize(frame, (width, height))
        image(frame_scaled, [0, 0], [width, height])

    # Handle "Take Photo" button
    if params.take_photo:
        video_processor.save_photo()
        params.take_photo = False

def key_pressed(key, modifier):
    if key == " ":
        toggle_gui()  # Show/hide GUI dynamically

def cleanup():
    global video_processor
    if video_processor:
        video_processor.release()

run()
