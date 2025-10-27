import cv2
import numpy as np
import time
import os

# --- Config ---
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, PARAMS)

CAM_INDEX = 0
RUN_SECONDS = 40
CSV_FILE = 'data/markers_behavior_log.csv'

EXPECTED_IDS = {0, 1, 2, 3, 4}

# Path colors (marker 0 fixed yellow)
PATH_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
]
marker_colors = {}

# --- Main Script ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit('Cannot open camera')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

marker_paths = {}
all_marker_data = []

start_time = time.time()
frame_count = 0
detected_frame_count = 0

_, background = cap.read()
if background is None:
    raise SystemExit('Cannot read a frame from the camera. Check camera connection.')
background = np.zeros_like(background)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_center = np.array([frame_width / 2, frame_height / 2])

# --- for plotting every 0.1s ---
last_plot_time = start_time
plot_interval = 0.1  # seconds

while (time.time() - start_time) < RUN_SECONDS:
    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1
    current_time = time.time()
    corners, ids, _ = DETECTOR.detectMarkers(frame)

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    current_frame_data = {'time': current_time - start_time}

    if ids is not None:
        ids = ids.flatten()
        for i, c in zip(ids, corners):
            c = c.reshape(-1, 2)
            center_px = c.mean(axis=0)

            # --- Coordinates relative to camera center ---
            relative_coords = center_px - camera_center

            # Draw marker on live frame
            cv2.circle(frame, (int(center_px[0]), int(center_px[1])), 20, (0, 255, 0), 2)
            cv2.circle(frame, (int(center_px[0]), int(center_px[1])), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{i} ({int(relative_coords[0])},{int(relative_coords[1])})",
                        (int(center_px[0]) + 10, int(center_px[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Save coordinates (relative)
            current_frame_data[f'marker_{i}_rel_x'] = relative_coords[0]
            current_frame_data[f'marker_{i}_rel_y'] = relative_coords[1]

            # Assign colors
            if i not in marker_colors:
                if i == 0:
                    marker_colors[i] = (0, 255, 255)  # Yellow for marker 0
                else:
                    marker_colors[i] = PATH_COLORS[len(marker_colors) % len(PATH_COLORS)]

            # Save paths
            if i not in marker_paths:
                marker_paths[i] = []
            marker_paths[i].append(relative_coords)

            # --- Plot every 0.1s ---
            if current_time - last_plot_time >= plot_interval:
                pt = tuple((center_px).astype(int))
                cv2.circle(background, pt, 3, marker_colors[i], -1)

    if current_time - last_plot_time >= plot_interval:
        last_plot_time = current_time

    all_marker_data.append(current_frame_data)

    final_frame = cv2.addWeighted(frame, 1, background, 1, 0)
    fps = frame_count / (time.time() - start_time)

    # Draw coordinate axis at camera center
    cv2.line(final_frame, (int(camera_center[0]), 0),
             (int(camera_center[0]), frame_height), (255, 0, 0), 2)
    cv2.line(final_frame, (0, int(camera_center[1])),
             (frame_width, int(camera_center[1])), (255, 0, 0), 2)

    cv2.putText(final_frame, f'Markers: {len(ids) if ids is not None else 0}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(final_frame, f'FPS: {fps:.2f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Markers Behavior', final_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save final background with paths
os.makedirs('data', exist_ok=True)
final_image_path = os.path.join('data', 'final_path_plot.png')
cv2.imwrite(final_image_path, background)
print(f"Final path plot saved to {final_image_path}")

# Save CSV
if all_marker_data:
    with open(CSV_FILE, 'w') as f:
        header = ['time']
        unique_markers = sorted(list(set(key.split('_')[1] for d in all_marker_data for key in d if key.startswith('marker'))))
        for marker_id in unique_markers:
            header.extend([f'{marker_id}_rel_x', f'{marker_id}_rel_y'])
        f.write(','.join(header) + '\n')

        for row_data in all_marker_data:
            row_str = [f'{row_data.get("time", "")}']
            for marker_id in unique_markers:
                row_str.extend([
                    f'{row_data.get(f"marker_{marker_id}_rel_x", "")}',
                    f'{row_data.get(f"marker_{marker_id}_rel_y", "")}'
                ])
            f.write(','.join(row_str) + '\n')
    print(f"Marker data saved to {CSV_FILE}")
