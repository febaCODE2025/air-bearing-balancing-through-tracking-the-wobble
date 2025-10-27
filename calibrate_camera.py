# --- File 2: calibrate_camera.py ---
import cv2
import numpy as np
import glob
import os

# Settings from the previous script
PATTERN_SIZE = (9, 6) # Inner corners
SQUARE_SIZE = 25.0 # mm (This needs to be the actual size of your squares)
IMG_DIR = 'data/calib_images/*.png'
OUT_FILE = 'calib/camera_intrinsics.npz'

# Prepare object points, like (0,0,0), (25,0,0), (50,0,0) ...., (200,125,0)
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(IMG_DIR)

if not images:
    raise SystemExit('No calibration images found. Please run take_calibration_photos.py first.')

# Process each image
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
    
    # If a pattern is found, refine it and add to the lists
    if ret:
        # Refine the corners for higher accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"Found corners in {os.path.basename(fname)}")
    else:
        print(f"Warning: Could not find corners in {os.path.basename(fname)}. This image will be skipped.")

if not objpoints:
    raise SystemExit("No valid chessboard images were found. Please check your photos and settings.")

# Calibrate the camera
# This is the core function that calculates the camera matrix and distortion coefficients.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the RMS reprojection error. A value less than 1.0 is generally considered good.
print('\n--- Calibration Results ---')
print(f'RMS reprojection error: {ret:.4f}')

if ret > 1.0:
    print("Warning: The reprojection error is high. Consider using more or better quality images.")

# Create the output directory and save the calibration data.
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.savez(OUT_FILE, mtx=mtx, dist=dist)

print(f'\nCalibration data saved to {OUT_FILE}')
print("You can now use this file in your main computer vision script.")
