#!/usr/bin/env python3
import os
import time
import pickle
import numpy as np

from extract_topics import McapTimeWindowExtractor
from imu_utils import IMUPreintegrator

# ================================
# CONFIGURATION
# ================================
MCAP_PATH       = "huntington.mcap"
CAMERA_TOPIC    = "/cam_sync/cam0/image_raw/compressed"
LIDAR_TOPIC     = "/ouster/points"
IMU_TOPIC       = "/vectornav/imu_uncompensated"

# Use the SAME window as you used for camera extraction
START_TIME = 10.0   # seconds from bag start
DURATION   = 5.0    # seconds

# Where to save the final densified LiDAR frames
OUTPUT_DIR      = "/home/aryaman/Huntington_Data/calib_cache"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "lidar_scans.pkl")

# LiDAR projection params (as before)
H, W = 128, 1024
FOV_UP, FOV_DOWN = 22.5, -22.5


# ================================
# LiDAR helper functions
# ================================
def project_to_range_image(points, height=128, width=1024):
    """Project 3D points to a (height, width, 3) grid in LiDAR coordinates."""
    total = height * width
    if points.shape[0] == total:
        return points.reshape(height, width, 3)

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

        u[valid_mask] = u_val.astype(int)
        v[valid_mask] = v_val.astype(int)

    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    img = np.zeros((height, width, 3), dtype=np.float32)
    valid = valid_mask & (v >= 0) & (v < height) & (u >= 0) & (u < width)
    img[v[valid], u[valid]] = points[valid]

    return img


def extract_edges(grid_points):
    """Return edge points where range jumps > 1.0 m between horizontal neighbours."""
    ranges = np.linalg.norm(grid_points, axis=2)
    diff = np.abs(np.diff(ranges, axis=1))
    diff = np.pad(diff, ((0, 0), (0, 1)), constant_values=0)
    return grid_points[diff > 1.0]


def apply_transform(points, T):
    """Apply 4x4 transform T to Nx3 points."""
    if len(points) == 0:
        return points
    ones = np.ones((len(points), 1), dtype=points.dtype)
    pts_hom = np.hstack((points, ones))  # (N,4)
    pts_out = (T @ pts_hom.T).T
    return pts_out[:, :3]


# ================================
# MAIN
# ================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading MCAP: {MCAP_PATH}")
    print(f"Time window: start={START_TIME}s  duration={DURATION}s")

    extractor = McapTimeWindowExtractor(
        mcap_path=MCAP_PATH,
        camera_topic=CAMERA_TOPIC,
        lidar_topic=LIDAR_TOPIC,
        imu_topic=IMU_TOPIC,
        start_time=START_TIME,
        duration=DURATION,
    )
    extractor.extract()

    print(f"\n[SUMMARY]")
    print(f"  Camera frames: {len(extractor.camera_frames)}")
    print(f"  LiDAR scans:   {len(extractor.lidar_scans)}")
    print(f"  IMU samples:   {len(extractor.imu_measurements)}")

    if len(extractor.imu_measurements) == 0:
        print("CRITICAL: No IMU data found, cannot motion-compensate LiDAR.")
        return

    # IMU preintegrator (sign of gravity should match imu_utils implementation)
    print("\nInitializing IMU preintegrator...")
    imu_integrator = IMUPreintegrator(gravity=np.array([0.0, 0.0, 9.81]))

    window = []         # stores last 3 frames: [{'t': ..., 'edges': ...}, ...]
    processed_data = [] # list of {'timestamp': float, 'points': (N,3) float32}

    total_scans = len(extractor.lidar_scans)
    print(f"\nProcessing {total_scans} LiDAR scans in calibration window...")

    t0 = time.time()

    for i, scan in enumerate(extractor.lidar_scans):
        if i % 10 == 0:
            print(f"  LiDAR scan {i}/{total_scans}")

        raw_points = scan["points"]          # (N,3)
        timestamp = scan["timestamp_sec"]    # float

        # 1. Feature extraction: range image + edge points
        grid = project_to_range_image(raw_points, H, W)
        edges = extract_edges(grid)
        if edges.size == 0:
            continue

        # 2. Push into sliding window
        window.append({"t": timestamp, "edges": edges})
        if len(window) > 3:
            window.pop(0)

        # 3. Once we have 3 frames, densify in the current frame coordinates
        if len(window) == 3:
            f_prev, f_curr, f_next = window[0], window[1], window[2]

            # IMU-based motion compensation:
            #  - bring prev -> curr
            T_p2c = imu_integrator.preintegrate(
                extractor.imu_measurements,
                f_prev["t"],
                f_curr["t"],
            )
            edges_prev_aligned = apply_transform(f_prev["edges"], T_p2c)

            #  - bring next -> curr
            T_c2n = imu_integrator.preintegrate(
                extractor.imu_measurements,
                f_curr["t"],
                f_next["t"],
            )
            T_n2c = np.linalg.inv(T_c2n)
            edges_next_aligned = apply_transform(f_next["edges"], T_n2c)

            # 4. Stack: densified, motion-corrected edges at f_curr time
            dense_corrected = np.vstack(
                [edges_prev_aligned, f_curr["edges"], edges_next_aligned]
            )

            frame_data = {
                "timestamp": float(f_curr["t"]),                 # keep key name 'timestamp'
                "points": dense_corrected.astype(np.float32),    # (N,3)
            }
            processed_data.append(frame_data)

    elapsed = time.time() - t0
    print(f"\nProcessing complete in {elapsed:.2f} s.")
    print(f"Generated {len(processed_data)} densified LiDAR frames.")

    print(f"\nSaving to {OUTPUT_FILENAME} ...")
    with open(OUTPUT_FILENAME, "wb") as f:
        pickle.dump(processed_data, f)

    print("Done.")


if __name__ == "__main__":
    main()

