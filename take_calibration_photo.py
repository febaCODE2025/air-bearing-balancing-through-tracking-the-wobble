# --- File 1: take_calibration_photos.py ---
import cv2
import os
import time

# --- Settings ---
# Requested resolution for a 2K camera. This will be the resolution used for calibration.
FRAME_WIDTH = 2560
FRAME_HEIGHT = 1440

# A standard display resolution to resize the frame for optimal viewing on most laptops.
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Directory to save calibration images
IMG_DIR = 'data/calib_images'
PATTERN_SIZE = (9, 6) # Inner corners of the chessboard pattern

# Create the directory if it doesn't exist
os.makedirs(IMG_DIR, exist_ok=True)

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit('Error: Could not open camera. Please check camera index or connection.')

# Attempt to set the desired resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Requested resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
print(f"Actual camera resolution: {int(actual_width)}x{int(actual_height)}")

print("\nInstructions:")
print(f"1. Move the chessboard to different positions and angles within the camera's view.")
print(f"2. Press 's' to save an image when the pattern is visible.")
print(f"3. Press 'q' to quit and stop taking pictures.")
print("A green border will appear around the image when the pattern is detected successfully.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Find the chessboard corners to provide visual feedback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found_corners, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

    display_frame = frame.copy()
    
    # Draw a green border if corners are detected, a red one otherwise.
    if found_corners:
        cv2.drawChessboardCorners(display_frame, PATTERN_SIZE, corners, found_corners)
        feedback_color = (0, 255, 0) # Green
    else:
        feedback_color = (0, 0, 255) # Red

    cv2.putText(display_frame, f"Good Images: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Resize for display on the screen
    display_frame = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow('Camera Calibration', display_frame)

    key = cv2.waitKey(1) & 0xFF
    
    # If the user presses 's' and corners are found, save the picture
    if key == ord('s') and found_corners:
        filename = os.path.join(IMG_DIR, f'calib_image_{count}.png')
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1
        time.sleep(0.5) # Add a small delay to avoid saving multiple images at once
    
    # If the user presses 'q', quit the program
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
