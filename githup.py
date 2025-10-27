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
CALIB_FILE = 'calib/camera_intrinsics.npz'
try:
    calib_data = np.load(CALIB_FILE)
    camera_matrix = calib_data['mtx']
    dist_coeffs = calib_data['dist']
    print("Camera calibration data loaded successfully.")
except FileNotFoundError:
    raise SystemExit('Error: Camera calibration file not found.')

# --- Physical Setup ---
R_REF = 120.0  # mm, radius of reference markers 1-4 from COR
CAM_INDEX = 0
RUN_SECONDS = 5

CSV_FILE_REAL_WORLD = 'data/markers_real_world_log.csv'
CSV_FILE_PIXEL = 'data/markers_pixel_log.csv'
FINAL_IMAGE_FILE_MM = 'data/final_path_plot_mm.png'
FINAL_IMAGE_FILE_PIXEL = 'data/final_path_plot_pixel.png'
CENTER_ID = 0

# Heights (in mm)
Z_CAM = 500    # Camera height above table (50 cm)
Z_TABLE = 360  # Table markers height below camera (36 cm)
Z_MARKER0 = 240 # Marker 0 height below camera (24 cm)

REF_COORDS_REAL = {
    1: np.array([+R_REF, 0.0]),    # +X
    2: np.array([0.0, +R_REF]),    # +Y
    3: np.array([-R_REF, 0.0]),    # -X
    4: np.array([0.0, -R_REF]),    # -Y
}

PATH_COLORS = {
    0: (0, 255, 255),
    1: (255, 0, 255),
    2: (0, 0, 255),
    3: (0, 255, 0),
    4: (0, 128, 255),
}

EXPECTED_IDS = {0, 1, 2, 3, 4}

def run_tracking():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit('Cannot open camera')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    marker_paths_pixel = {}
    marker_paths_mm = {}
    all_marker_data_real_world = []
    all_marker_data_pixel = []
    REF_COORDS_PIXELS = {}

    start_time = time.time()
    frame_count = 0

    _, frame_for_size = cap.read()
    if frame_for_size is None:
        raise SystemExit('Cannot read a frame from the camera. Check camera connection.')
    background_frame_pixel = np.zeros_like(frame_for_size)
    background_frame_mm = np.zeros_like(frame_for_size)

    while (time.time() - start_time) < RUN_SECONDS:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        corners, ids, _ = cv2.aruco.detectMarkers(frame, DICTIONARY, parameters=PARAMS)

        current_frame_data_pixel = {'time': time.time() - start_time}
        current_frame_data_real_world = {'time': time.time() - start_time}

        detected_ref_ids = []
        if ids is not None:
            ids_flat = ids.flatten()
            filtered_ids = []
            filtered_corners = []
            for i, c in zip(ids_flat, corners):
                if i in EXPECTED_IDS:
                    filtered_ids.append(i)
                    filtered_corners.append(c)
            ids = np.array(filtered_ids)
            corners = np.array(filtered_corners)

            if len(ids) > 0:
                ids_flat = ids.flatten()
                for i, c in zip(ids_flat, corners):
                    c = c.reshape(-1, 2)
                    center_px = c.mean(axis=0)

                    if i in REF_COORDS_REAL:
                        REF_COORDS_PIXELS[i] = center_px

                    if i not in marker_paths_pixel:
                        marker_paths_pixel[i] = []
                    if len(marker_paths_pixel[i]) > 0:
                        pt1 = tuple(marker_paths_pixel[i][-1].astype(int))
                        pt2 = tuple(center_px.astype(int))
                        path_color = PATH_COLORS.get(i, (255, 255, 255))
                        cv2.line(background_frame_pixel, pt1, pt2, path_color, 2)
                    marker_paths_pixel[i].append(center_px)
                    
                    current_frame_data_pixel[f'marker_{i}_px_x'] = center_px[0]
                    current_frame_data_pixel[f'marker_{i}_px_y'] = center_px[1]

                all_marker_data_pixel.append(current_frame_data_pixel)

                # Use at least 3 reference markers for transformation
                detected_ref_ids = [i for i in REF_COORDS_PIXELS if i in ids_flat]
                if len(detected_ref_ids) >= 3 and CENTER_ID in ids_flat:
                    src_pts = np.float32([REF_COORDS_PIXELS[i] for i in detected_ref_ids])
                    dst_pts = np.float32([REF_COORDS_REAL[i] for i in detected_ref_ids])
                    center_marker_corners = corners[np.where(ids_flat == CENTER_ID)[0][0]]
                    center_px = center_marker_corners.mean(axis=1)
                    undistorted_px = cv2.undistortPoints(np.array([center_px]), camera_matrix, dist_coeffs, P=camera_matrix)[0][0]

                    if len(detected_ref_ids) == 3:
                        M = cv2.getAffineTransform(src_pts.reshape(3,2), dst_pts.reshape(3,2))
                        real_world_coords = cv2.transform(undistorted_px.reshape(-1,1,2), M)[0][0]
                    else:
                        M, _ = cv2.findHomography(src_pts.reshape(-1,1,2), dst_pts.reshape(-1,1,2), cv2.RANSAC, 5.0)
                        real_world_coords = cv2.perspectiveTransform(undistorted_px.reshape(-1,1,2), M)[0][0]

                    x_real, y_real = real_world_coords[0], real_world_coords[1]
                    current_frame_data_real_world['marker_0_x_mm'] = x_real
                    current_frame_data_real_world['marker_0_y_mm'] = y_real

                    all_marker_data_real_world.append(current_frame_data_real_world)

                    if CENTER_ID not in marker_paths_mm:
                        marker_paths_mm[CENTER_ID] = []
                    mm_to_px_scale = background_frame_mm.shape[0] / (R_REF * 4)
                    center_mm_plot = np.array([x_real * mm_to_px_scale, y_real * mm_to_px_scale])
                    
                    if len(marker_paths_mm[CENTER_ID]) > 0:
                        pt1_mm = marker_paths_mm[CENTER_ID][-1]
                        pt2_mm = center_mm_plot
                        pt1_px = (int(pt1_mm[0] + background_frame_mm.shape[1]/2), int(pt1_mm[1] + background_frame_mm.shape[0]/2))
                        pt2_px = (int(pt2_mm[0] + background_frame_mm.shape[1]/2), int(pt2_mm[1] + background_frame_mm.shape[0]/2))
                        cv2.line(background_frame_mm, pt1_px, pt2_px, PATH_COLORS.get(CENTER_ID), 2)
                    marker_paths_mm[CENTER_ID].append(center_mm_plot)

        # Draw coordinate directions and marker labels
        combined_pixel_frame = cv2.addWeighted(frame, 1.0, background_frame_pixel, 1.0, 0)
        display_frame = cv2.resize(combined_pixel_frame, (1280, 720))
        h, w, _ = display_frame.shape

        # Draw axes with direction labels (+X, -X, +Y, -Y)
        cv2.line(display_frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 2) # Y axis
        cv2.line(display_frame, (0, h // 2), (w, h // 2), (255, 0, 0), 2) # X axis
        cv2.putText(display_frame, '+X', (w - 50, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(display_frame, '-X', (10, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(display_frame, '+Y', (w // 2 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(display_frame, '-Y', (w // 2 + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.circle(display_frame, (w // 2, h // 2), 5, (0, 255, 0), -1)
        
        # Draw marker labels
        for mid, path in marker_paths_pixel.items():
            if len(path) > 0:
                pt = tuple(path[-1].astype(int))
                cv2.putText(display_frame, f'M{mid}', pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, PATH_COLORS[mid], 2)

        fps = frame_count / (time.time() - start_time)
        feedback_y = 30
        cv2.putText(display_frame, f'FPS: {fps:.2f}', (10, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Live Tracking', display_frame)

        if cv2.waitKey(1) == 27:
            break
    
    os.makedirs(os.path.dirname(FINAL_IMAGE_FILE_PIXEL), exist_ok=True)
    cv2.imwrite(FINAL_IMAGE_FILE_PIXEL, background_frame_pixel)
    print(f"Final pixel path plot saved to {FINAL_IMAGE_FILE_PIXEL}")

    cap.release()
    cv2.destroyAllWindows()

    if all_marker_data_pixel:
        os.makedirs(os.path.dirname(CSV_FILE_PIXEL), exist_ok=True)
        fieldnames = sorted(list(set(key for d in all_marker_data_pixel for key in d)))
        with open(CSV_FILE_PIXEL, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_marker_data_pixel)
        print(f"\nAll pixel data saved to {CSV_FILE_PIXEL}")

    return all_marker_data_real_world, background_frame_mm


def analyze_data(data, background_mm):
    if not data:
        print("No data collected for analysis.")
        return

    marker_0_data = [d for d in data if 'marker_0_x_mm' in d]
    if len(marker_0_data) < 5:
        print("Not enough data points for Marker 0 to fit a circle.")
        return

    x_coords = np.array([d['marker_0_x_mm'] for d in marker_0_data])
    y_coords = np.array([d['marker_0_y_mm'] for d in marker_0_data])

    os.makedirs(os.path.dirname(CSV_FILE_REAL_WORLD), exist_ok=True)
    with open(CSV_FILE_REAL_WORLD, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'marker_0_x_mm', 'marker_0_y_mm'])
        writer.writeheader()
        writer.writerows(marker_0_data)
    print(f"\nReal-world marker data saved to {CSV_FILE_REAL_WORLD}")

    def calc_residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    x_init, y_init = np.mean(x_coords), np.mean(y_coords)
    r_init = np.mean(np.sqrt((x_coords - x_init)**2 + (y_coords - y_init)**2))
    initial_guess = [x_init, y_init, r_init]

    result = least_squares(calc_residuals, initial_guess, args=(x_coords, y_coords))
    x_path_center, y_path_center, radius_measured = result.x

    # --- Perspective correction for COM offset ---
    scale_factor = (Z_CAM - Z_TABLE) / (Z_CAM - Z_MARKER0)
    radius_true = radius_measured * scale_factor
    x_true = x_path_center * scale_factor
    y_true = y_path_center * scale_factor

    # --- Plotting with coordinate directions and marker labels ---
    pixel_plot_image = cv2.imread(FINAL_IMAGE_FILE_PIXEL)
    if pixel_plot_image is None:
        print(f"Error: Could not load pixel plot image at {FINAL_IMAGE_FILE_PIXEL}")
        return

    h_px, w_px, _ = pixel_plot_image.shape
    # Draw axes and direction labels
    cv2.line(pixel_plot_image, (w_px // 2, 0), (w_px // 2, h_px), (0, 0, 255), 2)
    cv2.line(pixel_plot_image, (0, h_px // 2), (w_px, h_px // 2), (0, 0, 255), 2)
    cv2.putText(pixel_plot_image, '+X', (w_px - 50, h_px // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(pixel_plot_image, '-X', (10, h_px // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(pixel_plot_image, '+Y', (w_px // 2 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(pixel_plot_image, '-Y', (w_px // 2 + 10, h_px - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.circle(pixel_plot_image, (w_px // 2, h_px // 2), 5, (0, 255, 0), -1)

    # Draw correction vector arrow
    scale_px = h_px / (R_REF * 4)
    center_px_x = int(w_px / 2 + x_true * scale_px)
    center_px_y = int(h_px / 2 - y_true * scale_px)
    cv2.circle(pixel_plot_image, (center_px_x, center_px_y), 6, (255, 0, 0), -1)
    correction_vector_end_x = int(w_px / 2 - x_true * scale_px)
    correction_vector_end_y = int(h_px / 2 + y_true * scale_px)
    cv2.arrowedLine(pixel_plot_image, (center_px_x, center_px_y), (correction_vector_end_x, correction_vector_end_y), (255,255,255), 5)

    cv2.imwrite(FINAL_IMAGE_FILE_PIXEL, pixel_plot_image)
    print(f"\nFinal pixel analysis plot saved to {FINAL_IMAGE_FILE_PIXEL}")

    # --- Plotting the results on the mm plot ---
    final_image = background_mm.copy()
    h, w, _ = final_image.shape
    mm_to_px_scale = h / (R_REF * 4)
    center_px_x = int(w / 2 + x_true * mm_to_px_scale)
    center_px_y = int(h / 2 - y_true * mm_to_px_scale)
    cv2.circle(final_image, (center_px_x, center_px_y), 5, (255, 0, 0), -1)
    radius_px = int(radius_true * mm_to_px_scale)
    cv2.circle(final_image, (center_px_x, center_px_y), radius_px, (255, 0, 255), 2)
    correction_vector_end_x = int(w / 2 - x_true * mm_to_px_scale)
    correction_vector_end_y = int(h / 2 + y_true * mm_to_px_scale)
    cv2.arrowedLine(final_image, (center_px_x, center_px_y), (correction_vector_end_x, correction_vector_end_y), (255,255,255), 5)
    cv2.line(final_image, (w // 2, 0), (w // 2, h), (0, 0, 255), 2)
    cv2.line(final_image, (0, h // 2), (w, h // 2), (0, 0, 255), 2)
    cv2.putText(final_image, '+X', (w - 50, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(final_image, '-X', (10, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(final_image, '+Y', (w // 2 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(final_image, '-Y', (w // 2 + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.circle(final_image, (w // 2, h // 2), 5, (0, 255, 0), -1
    )
    cv2.imwrite(FINAL_IMAGE_FILE_MM, final_image)
    print(f"\nFinal analysis plot saved to {FINAL_IMAGE_FILE_MM}")

    print("\n--- Analysis Results ---")
    print(f"Fitted Circle Center (Corrected): ({x_true:.2f} mm, {y_true:.2f} mm)")
    print(f"Fitted Circle Radius (Corrected): {radius_true:.2f} mm")

    correction_vector = np.array([x_true, y_true])
    magnitude = np.linalg.norm(correction_vector)
    direction = np.degrees(np.arctan2(y_true, x_true))

    print(f"\nCorrection Vector (Corrected):")
    print(f"  Magnitude: {magnitude:.2f} mm")
    print(f"  Direction: {direction:.2f} degrees (from +X axis)")
    print("Directions: +X is right, +Y is up (on plot). Move trim weight in this direction by the indicated amount.")

if __name__ == '__main__':
    real_world_data, mm_background = run_tracking()
    analyze_data(real_world_data, mm_background)