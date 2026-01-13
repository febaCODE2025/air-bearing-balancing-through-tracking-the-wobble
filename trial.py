import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import csv
from scipy.optimize import least_squares

# Config
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, PARAMS)

RUN_SECONDS = 10
R_REF = 120.0  # mm

PATH_COLORS = {
    0: (0, 255, 255),   # Yellow
    1: (255, 0, 255),   # Magenta
    2: (0, 0, 255),     # Red
    3: (0, 255, 0),     # Green
    4: (255, 128, 0),   # Orange
}

# Initialize RealSense with higher resolution and framerate
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)  # 60 FPS
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

print("Starting RealSense at 60 FPS...")
pipeline.start(config)

# Get depth parameters
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics

print(f"Depth scale: {depth_scale} (1 unit = {depth_scale*1000:.3f} mm)")
print(f"Camera intrinsics: fx={depth_intrin.fx:.2f}, fy={depth_intrin.fy:.2f}")
print(f"Resolution: {depth_intrin.width}x{depth_intrin.height}")

align = rs.align(rs.stream.color)

# CSV file setup
os.makedirs('data', exist_ok=True)
csv_file = open('data/markers_depth_log.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp_sec', 'marker_id', 
                     'pixel_x', 'pixel_y', 
                     'depth_mm', 'x_3d_mm', 'y_3d_mm', 'z_3d_mm'])

# Storage
marker_data = {i: {'positions_3d': [], 'positions_2d': [], 'depths': []} for i in range(5)}
background = None

# Real-time depth display
current_depths = {i: 0.0 for i in range(5)}

start_time = time.time()
frame_count = 0
detection_count = 0

print(f"\n=== Tracking for {RUN_SECONDS} seconds ===")
print("Press 'q' to stop early")
print("CSV file: data/markers_depth_log.csv")
print("Units: All depths and 3D coordinates in MILLIMETERS (mm)\n")

try:
    while (time.time() - start_time) < RUN_SECONDS:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        frame_count += 1
        elapsed = time.time() - start_time
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Initialize background
        if background is None:
            background = np.zeros_like(color_image)
        
        # Detect markers
        corners, ids, _ = DETECTOR.detectMarkers(color_image)
        
        if ids is not None:
            ids = ids.flatten()
            detection_count += len(ids)
            
            for marker_id, corner in zip(ids, corners):
                if marker_id not in marker_data:
                    continue
                
                # Get marker center in pixels
                corner_pts = corner.reshape(-1, 2)
                center_px = corner_pts.mean(axis=0)
                x_px, y_px = int(center_px[0]), int(center_px[1])
                
                # Get depth with 7x7 averaging for better accuracy
                depth_values = []
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        px = min(max(x_px + dx, 0), depth_intrin.width - 1)
                        py = min(max(y_px + dy, 0), depth_intrin.height - 1)
                        d = depth_frame.get_distance(px, py)
                        if d > 0.1:  # Filter out very close/invalid readings
                            depth_values.append(d)
                
                if len(depth_values) > 10:  # Need at least 10 valid points
                    depth = np.median(depth_values)  # Median for robustness
                    depth_mm = depth * 1000  # Convert to millimeters
                    
                    # Update real-time display
                    current_depths[marker_id] = depth_mm
                    
                    # Convert to 3D coordinates
                    x_3d, y_3d, z_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, [x_px, y_px], depth
                    )
                    
                    # Convert to millimeters
                    x_3d_mm = x_3d * 1000
                    y_3d_mm = y_3d * 1000
                    z_3d_mm = z_3d * 1000
                    
                    # Store data
                    marker_data[marker_id]['positions_3d'].append([x_3d_mm, y_3d_mm, z_3d_mm])
                    marker_data[marker_id]['positions_2d'].append([x_px, y_px])
                    marker_data[marker_id]['depths'].append(depth_mm)
                    
                    # Write to CSV immediately
                    csv_writer.writerow([
                        f'{elapsed:.3f}',
                        marker_id,
                        x_px, y_px,
                        f'{depth_mm:.2f}',
                        f'{x_3d_mm:.2f}',
                        f'{y_3d_mm:.2f}',
                        f'{z_3d_mm:.2f}'
                    ])
                    
                    # Draw path
                    if len(marker_data[marker_id]['positions_2d']) > 1:
                        pt1 = tuple(map(int, marker_data[marker_id]['positions_2d'][-2]))
                        pt2 = tuple(map(int, marker_data[marker_id]['positions_2d'][-1]))
                        cv2.line(background, pt1, pt2, PATH_COLORS[marker_id], 2)
                    
                    # Draw marker
                    color = PATH_COLORS[marker_id]
                    cv2.circle(color_image, (x_px, y_px), 8, color, 2)
                    cv2.circle(color_image, (x_px, y_px), 3, color, -1)
                    
                    # Show ID and REAL-TIME DEPTH
                    text = f"ID{marker_id}"
                    cv2.putText(color_image, text, (x_px+12, y_px-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    text_depth = f"{depth_mm:.1f}mm"
                    cv2.putText(color_image, text_depth, (x_px+12, y_px), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Combine with paths
        combined = cv2.addWeighted(color_image, 0.7, background, 0.3, 0)
        
        # Draw large depth display panel (RIGHT SIDE)
        panel_x = 450
        y_pos = 30
        
        cv2.rectangle(combined, (panel_x-10, 10), (630, 180), (0, 0, 0), -1)
        cv2.putText(combined, "REAL-TIME DEPTH", (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
        
        for marker_id in range(5):
            color = PATH_COLORS[marker_id]
            depth_val = current_depths[marker_id]
            text = f"M{marker_id}: {depth_val:.1f} mm"
            cv2.putText(combined, text, (panel_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25
        
        # Info panel (TOP LEFT)
        cv2.rectangle(combined, (5, 5), (250, 120), (0, 0, 0), -1)
        y_pos = 25
        cv2.putText(combined, f"Time: {elapsed:.1f}/{RUN_SECONDS}s", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 22
        cv2.putText(combined, f"FPS: {frame_count/elapsed:.1f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 22
        
        for marker_id in range(5):
            count = len(marker_data[marker_id]['positions_3d'])
            color = PATH_COLORS[marker_id]
            text = f"M{marker_id}: {count}"
            cv2.putText(combined, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_pos += 18
        
        cv2.imshow('RealSense Depth Tracking', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
    
    csv_file.close()
    cv2.destroyAllWindows()
    
    # ===== ANALYSIS =====
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {frame_count/elapsed:.1f}")
    print(f"Total detections: {detection_count}")
    
    for marker_id in range(5):
        count = len(marker_data[marker_id]['positions_3d'])
        print(f"\nMarker {marker_id}: {count} detections")
        
        if count > 0:
            depths = marker_data[marker_id]['depths']
            print(f"  Depth (mm): min={min(depths):.1f}, max={max(depths):.1f}, avg={np.mean(depths):.1f}")
            
            positions = np.array(marker_data[marker_id]['positions_3d'])
            print(f"  Position (mm):")
            print(f"    X: min={positions[:,0].min():.1f}, max={positions[:,0].max():.1f}")
            print(f"    Y: min={positions[:,1].min():.1f}, max={positions[:,1].max():.1f}")
            print(f"    Z: min={positions[:,2].min():.1f}, max={positions[:,2].max():.1f}")
    
    # Marker 0 analysis
    if len(marker_data[0]['positions_3d']) >= 10:
        print("\n" + "="*60)
        print("MARKER 0 ANALYSIS (CoM Detection)")
        print("="*60)
        
        trajectory = np.array(marker_data[0]['positions_3d'])
        
        # Fit circle
        def calc_residuals(params, x, y):
            xc, yc, r = params
            return np.sqrt((x - xc)**2 + (y - yc)**2) - r
        
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        x_init, y_init = np.mean(x_coords), np.mean(y_coords)
        r_init = np.std(np.sqrt((x_coords - x_init)**2 + (y_coords - y_init)**2))
        
        result = least_squares(calc_residuals, [x_init, y_init, max(r_init, 0.1)], 
                              args=(x_coords, y_coords))
        
        x_center, y_center, radius = result.x
        z_center = np.mean(trajectory[:, 2])
        
        print(f"\nFitted Circle Center: ({x_center:.2f}, {y_center:.2f}, {z_center:.2f}) mm")
        print(f"Fitted Circle Radius: {radius:.2f} mm")
        
        # Calculate CoR from reference markers
        ref_detected = [i for i in [1, 2, 3, 4] if len(marker_data[i]['positions_3d']) > 0]
        
        if len(ref_detected) >= 3:
            ref_positions = []
            for i in ref_detected:
                avg_pos = np.mean(marker_data[i]['positions_3d'], axis=0)
                ref_positions.append(avg_pos)
            
            cor_3d = np.mean(ref_positions, axis=0)
            
            print(f"\nCenter of Rotation (CoR): ({cor_3d[0]:.2f}, {cor_3d[1]:.2f}, {cor_3d[2]:.2f}) mm")
            print(f"  (Calculated from markers: {ref_detected})")
            
            x_offset = x_center - cor_3d[0]
            y_offset = y_center - cor_3d[1]
            z_offset = z_center - cor_3d[2]
            
            print(f"\nCoM Offset from CoR:")
            print(f"  X: {x_offset:.2f} mm")
            print(f"  Y: {y_offset:.2f} mm")
            print(f"  Z: {z_offset:.2f} mm")
            
            magnitude = np.sqrt(x_offset**2 + y_offset**2)
            direction = np.degrees(np.arctan2(y_offset, x_offset))
            
            print(f"\nCorrection Vector (XY plane):")
            print(f"  Magnitude: {magnitude:.2f} mm")
            print(f"  Direction: {direction:.2f} degrees")
        else:
            print(f"\n⚠️  WARNING: Only {len(ref_detected)} reference markers detected")
            print(f"   Detected: {ref_detected}")
            print("   Need at least 3 reference markers for CoR calculation")
    
    print("\n" + "="*60)
    print("DATA SAVED")
    print("="*60)
    print("File: data/markers_depth_log.csv")
    print("\nCSV Columns:")
    print("  - timestamp_sec: Time in seconds")
    print("  - marker_id: Marker ID (0-4)")
    print("  - pixel_x, pixel_y: Pixel coordinates")
    print("  - depth_mm: Depth in MILLIMETERS")
    print("  - x_3d_mm, y_3d_mm, z_3d_mm: 3D position in MILLIMETERS")
    print("\nAll units are MILLIMETERS (mm)")

finally:
    pipeline.stop()
    if not csv_file.closed:
        csv_file.close()
    print("\n" + "="*60)
    print("Tracking complete!")
    print("="*60)