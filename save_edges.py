import numpy as np
import pickle
import time
import os
from extract_topics import McapTimeWindowExtractor
from imu_utils import IMUPreintegrator

## CONFIGURATION ---
MCAP_PATH = "bag2_forsyth_street_all.mcap" 
OUTPUT_FILENAME = "lidar_scans.pkl"

LIDAR_TOPIC = "/ouster/points"
# We don't strictly need camera topic here, but the extractor expects it
CAMERA_TOPIC = "/zed2i/zed_node/left_raw/image_raw_color" 
IMU_TOPIC = "/vectornav/imu_uncompensated"

# LiDAR Parameters
H, W = 128, 1024 # LiDAR Resolution
FOV_UP, FOV_DOWN = 22.5, -22.5 # Vertical FOV

## Projection of LiDAR points to LiDAR's Dimensions
def project_to_range_image(points, height=128, width=1024):
    """ Robust Grid Restoration """
    total = height * width
    if points.shape[0] == total: return points.reshape(height, width, 3)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    valid_mask = r > 0.001
    
    u = np.zeros_like(x, dtype=int)
    v = np.zeros_like(x, dtype=int)
    
    if np.any(valid_mask):
        x_v, y_v, z_v, r_v = x[valid_mask], y[valid_mask], z[valid_mask], r[valid_mask]
        yaw = np.arctan2(y_v, x_v)
        pitch = np.arcsin(np.clip(z_v / r_v, -1, 1))
        u_val = 0.5 * (yaw / np.pi + 1.0) * width
        fov_range = np.deg2rad(FOV_UP - FOV_DOWN)
        v_val = (1.0 - (pitch - np.deg2rad(FOV_DOWN)) / fov_range) * height
        u[valid_mask], v[valid_mask] = u_val.astype(int), v_val.astype(int)

    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    img = np.zeros((height, width, 3), dtype=np.float32)
    valid = valid_mask & (v >= 0) & (v < height) & (u >= 0) & (u < width)
    img[v[valid], u[valid]] = points[valid]
    return img

## Extract Edges from the LiDAR Map
def extract_edges(grid_points):
    """ Extract Edges > 1.0m jump """
    ranges = np.linalg.norm(grid_points, axis=2)
    diff = np.abs(np.diff(ranges, axis=1))
    diff = np.pad(diff, ((0,0), (0,1)), constant_values=0)
    return grid_points[diff > 1.0]

## Apply 4x4 Transformation to Point Cloud (Initialize Homogeneous Coordinates)
def apply_transform(points, T):
    if len(points) == 0: return points
    ones = np.ones((len(points), 1))
    pts_hom = np.hstack((points, ones))
    return (T @ pts_hom.T).T[:, :3]

def main():
    print(f"Loading MCAP: {MCAP_PATH} ...")
    extractor = McapTimeWindowExtractor(
        mcap_path=MCAP_PATH, camera_topic=CAMERA_TOPIC,
        lidar_topic=LIDAR_TOPIC, imu_topic=IMU_TOPIC,
        start_time=0.0, duration=None 
    )
    extractor.extract()
    
    # Check IMU
    if len(extractor.imu_measurements) == 0:
        print("CRITICAL ERROR: No IMU data found! Cannot perform motion compensation.")
        return

    # Initialize Integrator
    print("Initializing IMU Integrator...")
    imu_integrator = IMUPreintegrator(gravity=np.array([0, 0, 9.81])) # Ensure sign is correct!
    
    window = [] # Buffer
    processed_data = [] # The list we will save
    
    total_scans = len(extractor.lidar_scans)
    print(f"Processing {total_scans} scans for export...")

    start_time = time.time()

    for i, scan in enumerate(extractor.lidar_scans):
        # Progress bar
        if i % 10 == 0:
            print(f"Processing frame {i}/{total_scans} ...")

        raw_points = scan['points']
        timestamp = scan['timestamp_sec']
        
        # 1. Feature Extraction
        grid = project_to_range_image(raw_points, H, W)
        edges = extract_edges(grid)
        
        if len(edges) == 0: continue
        
        # 2. Buffer
        window.append({'t': timestamp, 'edges': edges})
        if len(window) > 3: window.pop(0)

        # 3. Densify & Correct
        if len(window) == 3:
            f_prev, f_curr, f_next = window[0], window[1], window[2]
            
            # 1. Align previous scan to current scan with IMU motion compensation
            T_p2c = imu_integrator.preintegrate(extractor.imu_measurements, f_prev['t'], f_curr['t'])
            edges_prev_aligned = apply_transform(f_prev['edges'], T_p2c)
            
            # 2. Align next scan to current scan with IMU motion compensation
            T_c2n = imu_integrator.preintegrate(extractor.imu_measurements, f_curr['t'], f_next['t'])
            T_n2c = np.linalg.inv(T_c2n)
            edges_next_aligned = apply_transform(f_next['edges'], T_n2c)
            
            # 3. Stack (Densified Features)
            dense_corrected = np.vstack([edges_prev_aligned, f_curr['edges'], edges_next_aligned])
            
            # 4. Save to List
            frame_data = {
                'timestamp': f_curr['t'],
                'points': dense_corrected.astype(np.float32) # Save as float32 to save space
            }
            processed_data.append(frame_data)

    elapsed = time.time() - start_time
    print(f"\nProcessing Complete in {elapsed:.2f}s.")
    print(f"Generated {len(processed_data)} densified frames.")
    
    # 5. Export to Pickle
    print(f"Saving to {OUTPUT_FILENAME} ...")
    with open(OUTPUT_FILENAME, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Saved {OUTPUT_FILENAME} successfully.")

if __name__ == "__main__":
    main()