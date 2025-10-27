import cv2
import numpy as np
import time
import os
import csv
from scipy.optimize import least_squares

# --- Aruco Marker Configuration ---
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
PARAMS = cv2.aruco.DetectorParameters()

# --- Calibration Settings ---
# Load camera calibration data. Ensure this file exists.
CALIB_FILE = 'calib/camera_intrinsics.npz'
try:
    calib_data = np.load(CALIB_FILE)
    camera_matrix = calib_data['mtx']
    dist_coeffs = calib_data['dist']
    print("Camera calibration data loaded successfully.")
except FileNotFoundError:
    raise SystemExit('Error: Camera calibration file not found. Please run the calibration script first.')

# --- Physical Setup Settings ---
R_REF = 120.0  # mm, the radius of your reference markers 1-4
CAM_INDEX = 0
RUN_SECONDS = 90
CSV_FILE_REAL_WORLD = 'data/markers_real_world_log.csv'
CSV_FILE_PIXEL = 'data/markers_pixel_log.csv'
FINAL_IMAGE_FILE = 'data/final_path_plot.png'
CENTER_ID = 0

# The physical coordinates of the reference markers on the table.
# These points define our real-world coordinate system.
REF_COORDS_PIXELS = {}
REF_COORDS_REAL = {
    1: np.array([+R_REF, 0.0]),
    2: np.array([0.0, +R_REF]),
    3: np.array([-R_REF, 0.0]),
    4: np.array([0.0, -R_REF]),
}

# --- The list of colors for each marker path (BGR format).
PATH_COLORS = {
    0: (0, 255, 255),  # Yellow for Marker 0
    1: (255, 0, 255),  # Purple for Marker 1
    2: (0, 0, 255),    # Red for Marker 2
    3: (0, 255, 0),    # Green for Marker 3
    4: (0, 128, 255),  # Orange for Marker 4
}

# --- Main Script ---
def run_tracking():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit('Cannot open camera')

    # Use the resolution that the camera was calibrated with.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    marker_paths = {}
    all_marker_data_real_world = []
    all_marker_data_pixel = []
    
    start_time = time.time()
    frame_count = 0

    # Create a persistent background image to draw paths on
    _, background_frame = cap.read()
    if background_frame is None:
        raise SystemExit('Cannot read a frame from the camera. Check camera connection.')
    background_frame = np.zeros_like(background_frame)
    
    while (time.time() - start_time) < RUN_SECONDS:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        corners, ids, _ = cv2.aruco.detectMarkers(frame, DICTIONARY, parameters=PARAMS)

        current_frame_data_pixel = {'time': time.time() - start_time}
        current_frame_data_real_world = {'time': time.time() - start_time}

        if ids is not None:
            ids_flat = ids.flatten()
            for i, c in zip(ids_flat, corners):
                c = c.reshape(-1, 2)
                center_px = c.mean(axis=0)

                # Store the pixel coordinates of the reference markers for homography
                if i in REF_COORDS_REAL:
                    REF_COORDS_PIXELS[i] = center_px

                # Append the pixel data to the marker paths for drawing
                if i not in marker_paths:
                    marker_paths[i] = []
                marker_paths[i].append(center_px)
                
                # Append all pixel data for logging
                current_frame_data_pixel[f'marker_{i}_px_x'] = center_px[0]
                current_frame_data_pixel[f'marker_{i}_px_y'] = center_px[1]

            all_marker_data_pixel.append(current_frame_data_pixel)

            # Check if all four reference markers are found to calculate Homography.
            if len(REF_COORDS_PIXELS) == 4 and CENTER_ID in ids_flat:
                # Prepare points for homography calculation.
                src_pts = np.float32([REF_COORDS_PIXELS[1], REF_COORDS_PIXELS[2], REF_COORDS_PIXELS[3], REF_COORDS_PIXELS[4]]).reshape(-1, 1, 2)
                dst_pts = np.float32([REF_COORDS_REAL[1], REF_COORDS_REAL[2], REF_COORDS_REAL[3], REF_COORDS_REAL[4]]).reshape(-1, 1, 2)
                
                # Calculate the homography matrix
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Now, transform the center marker (Marker 0) to real-world coordinates.
                center_marker_corners = corners[np.where(ids_flat == CENTER_ID)[0][0]]
                center_px = center_marker_corners.mean(axis=1)

                # Undistort the pixel coordinates
                undistorted_px = cv2.undistortPoints(np.array([center_px]), camera_matrix, dist_coeffs, P=camera_matrix)[0][0]
                
                # Corrected call to perspectiveTransform
                real_world_coords = cv2.perspectiveTransform(undistorted_px.reshape(-1, 1, 2), M)[0][0]

                x_real, y_real = real_world_coords[0], real_world_coords[1]
                current_frame_data_real_world['marker_0_x_mm'] = x_real
                current_frame_data_real_world['marker_0_y_mm'] = y_real
                
                # Append the real-world data for analysis
                all_marker_data_real_world.append(current_frame_data_real_world)
        
        # --- Path Plotting on Background Frame ---
        if ids is not None:
            for i in ids.flatten():
                if i in marker_paths and len(marker_paths[i]) > 1:
                    pt1 = tuple(marker_paths[i][-2].astype(int))
                    pt2 = tuple(marker_paths[i][-1].astype(int))
                    path_color = PATH_COLORS.get(i, (255, 255, 255)) # Default to white
                    cv2.line(background_frame, pt1, pt2, path_color, 2)

        # Combine the live frame with the persistent background.
        combined_frame = cv2.addWeighted(frame, 1.0, background_frame, 1.0, 0)
        
        # Draw a smaller, displayable frame with the coordinate system on top.
        display_frame = cv2.resize(combined_frame, (1280, 720))
        h, w, _ = display_frame.shape
        cv2.line(display_frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)  # Blue Y-axis
        cv2.line(display_frame, (0, h // 2), (w, h // 2), (255, 0, 0), 2)  # Blue X-axis
        cv2.circle(display_frame, (w // 2, h // 2), 5, (0, 255, 0), -1)

        # --- Display Feedback ---
        fps = frame_count / (time.time() - start_time)
        feedback_y = 30
        cv2.putText(display_frame, f'FPS: {fps:.2f}', (10, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        feedback_y += 30
        cv2.putText(display_frame, f'Markers Detected: {len(ids) if ids is not None else 0}', (10, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        feedback_y += 30
        if CENTER_ID in marker_paths and len(marker_paths[CENTER_ID]) > 0:
            last_pos = marker_paths[CENTER_ID][-1]
            cv2.putText(display_frame, f'Marker 0 Pos: ({last_pos[0]:.2f}, {last_pos[1]:.2f})', (10, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if len(REF_COORDS_PIXELS) == 4:
            cv2.putText(display_frame, 'Reference Markers Calibrated', (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, f'Looking for ref. markers ({len(REF_COORDS_PIXELS)}/4)', (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
        cv2.imshow('Markers Behavior', display_frame)

        if cv2.waitKey(1) == 27:
            break
    
    # Save the final background for later use
    os.makedirs(os.path.dirname(FINAL_IMAGE_FILE), exist_ok=True)
    temp_background_path = 'data/temp_background.png'
    cv2.imwrite(temp_background_path, background_frame)
    cap.release()
    cv2.destroyAllWindows()
    
    # Save all pixel data for debugging
    if all_marker_data_pixel:
        os.makedirs(os.path.dirname(CSV_FILE_PIXEL), exist_ok=True)
        # Collect all unique field names
        fieldnames = sorted(list(set(key for d in all_marker_data_pixel for key in d)))
        with open(CSV_FILE_PIXEL, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_marker_data_pixel)
        print(f"\nAll pixel data saved to {CSV_FILE_PIXEL}")

    return all_marker_data_real_world

# --- Analysis & Circle Fitting ---
def analyze_data(data):
    if not data:
        print("No data collected for analysis.")
        return

    # Extract marker 0's data
    marker_0_data = [d for d in data if 'marker_0_x_mm' in d]
    if len(marker_0_data) < 5:
        print("Not enough data points for Marker 0 to fit a circle.")
        return

    # Extract x and y coordinates from the collected data.
    x_coords = np.array([d['marker_0_x_mm'] for d in marker_0_data])
    y_coords = np.array([d['marker_0_y_mm'] for d in marker_0_data])

    # Save data to CSV file.
    os.makedirs(os.path.dirname(CSV_FILE_REAL_WORLD), exist_ok=True)
    with open(CSV_FILE_REAL_WORLD, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'marker_0_x_mm', 'marker_0_y_mm'])
        writer.writeheader()
        writer.writerows(marker_0_data)
    print(f"\nReal-world marker data saved to {CSV_FILE_REAL_WORLD}")

    # Define the objective function for least squares circle fitting.
    def calc_residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    # Use a dummy initial guess for the circle center and radius.
    x_init, y_init = np.mean(x_coords), np.mean(y_coords)
    r_init = np.mean(np.sqrt((x_coords - x_init)**2 + (y_coords - y_init)**2))
    initial_guess = [x_init, y_init, r_init]

    # Perform least-squares optimization.
    result = least_squares(calc_residuals, initial_guess, args=(x_coords, y_coords))
    
    # Extract the fitted parameters.
    x_path_center, y_path_center, radius = result.x
    
    # --- Plotting the results on the final image ---
    # Load the background image with the paths.
    temp_background_path = 'data/temp_background.png'
    final_image = cv2.imread(temp_background_path)
    
    # Get image dimensions to convert mm to pixel coordinates for plotting.
    h, w, _ = final_image.shape
    scale_factor = w / (R_REF * 2) # A simple scale factor to convert mm to pixels.
    
    # Plot the Fitted Circle Center (blue).
    center_px_x = int(w / 2 + x_path_center * scale_factor)
    center_px_y = int(h / 2 + y_path_center * scale_factor)
    cv2.circle(final_image, (center_px_x, center_px_y), 5, (255, 0, 0), -1)
    
    # Plot the Fitted Circle Radius (pink arrow).
    radius_px = int(radius * scale_factor)
    cv2.circle(final_image, (center_px_x, center_px_y), radius_px, (255, 0, 255), 2)
    
    # Plot the Correction Vector (thick white arrow).
    correction_vector_end_x = int(w / 2 - x_path_center * scale_factor)
    correction_vector_end_y = int(h / 2 - y_path_center * scale_factor)
    cv2.arrowedLine(final_image, (center_px_x, center_px_y), (correction_vector_end_x, correction_vector_end_y), (255, 255, 255), 5)
    
    # --- Plot the coordinate system ---
    cv2.line(final_image, (w // 2, 0), (w // 2, h), (0, 0, 255), 2) # Red Y-axis
    cv2.line(final_image, (0, h // 2), (w, h // 2), (0, 0, 255), 2) # Red X-axis
    cv2.circle(final_image, (w // 2, h // 2), 5, (0, 255, 0), -1) # Green circle at the origin

    cv2.imwrite(FINAL_IMAGE_FILE, final_image)
    print(f"\nFinal analysis plot saved to {FINAL_IMAGE_FILE}")

    print("\n--- Analysis Results ---")
    print(f"Fitted Circle Center: ({x_path_center:.2f} mm, {y_path_center:.2f} mm)")
    print(f"Fitted Circle Radius: {radius:.2f} mm")

    correction_vector = np.array([x_path_center, y_path_center])
    magnitude = np.linalg.norm(correction_vector)
    direction = np.degrees(np.arctan2(y_path_center, x_path_center))

    print(f"\nCorrection Vector:")
    print(f"  Magnitude: {magnitude:.2f} mm")
    print(f"  Direction: {direction:.2f} degrees (from positive X-axis)")
    print("\nNext Step: Move the trim weights in the direction of this vector to a distance proportional to its magnitude.")


if __name__ == '__main__':
    all_data = run_tracking()
    analyze_data(all_data)