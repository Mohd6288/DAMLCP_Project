import cv2
from mediapipe_tracking import apply_hand_tracking, apply_face_detection
from pix2pix import generate_pix2pix

class VideoProcessor:
    def __init__(self, params):
        self.params = params
        self.vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

    def process_frame(self):
        ret, frame = self.vid.read()
        if not ret:
            print("Failed to capture frame.")
            return None

        # Convert to grayscale and apply smoothing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smoothed_frame = cv2.GaussianBlur(gray_frame, (0, 0), self.params.smoothing)

        # Apply Canny edge detection
        edges = cv2.Canny(smoothed_frame, self.params.threshold1, self.params.threshold2)

        # Optional: Face detection
        if self.params.face_detection:
            frame = apply_face_detection(frame)

        # Optional: Hand tracking
        if self.params.hand_tracking:
            frame = apply_hand_tracking(frame)

        # Combine edges with the frame and pass to Pix2Pix
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        processed_frame = cv2.addWeighted(frame, 0.7, edges_rgb, 0.3, 0)
        return generate_pix2pix(processed_frame)

    def save_photo(self):
        ret, frame = self.vid.read()
        if ret:
            filename = f"photo_{cv2.getTickCount()}.png"
            cv2.imwrite(filename, frame)
            print(f"Photo saved as {filename}")

    def release(self):
        self.vid.release()
