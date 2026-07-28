# Camera-LiDAR Temporal Calibration

**By Group 1** - a joint project for camera-LiDAR sensor calibration on the Hunter robot.

## Overview

Camera and LiDAR sensors on a mobile robot are rarely perfectly time-synchronized, and even a small timing offset degrades any downstream sensor fusion. This project builds an offline calibration pipeline that estimates and corrects for this offset using a cross-modal edge-alignment metric, rather than manual correspondence labeling.

Based on ["Temporal and Spatial Online Integrated Calibration for Camera and LiDAR"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9921858).

## Methodology

1. **Camera edge pipeline** - Canny edges are extracted from each camera frame, then a normalized distance transform is computed over the inverted edge map, giving every pixel a value representing its distance to the nearest edge.
2. **Cross-modal scoring** - LiDAR points are projected into the camera frame using the current extrinsic transform and intrinsics, then each projected point is scored via a Gaussian falloff on its distance-transform value (`exp(-d²/2σ²)`), so points landing near a real image edge score higher. Per-scan scores are summed, then averaged across all LiDAR scans in the calibration window to form the global objective.
3. **Optimization** - Stage 1 optimizes only the temporal offset (tau) via Powell's method, minimizing the negative mean alignment score, converging in **0.293 s**. Stage 2 extends this to a joint 7-DOF optimization over temporal offset and extrinsic parameters (rotation + translation between camera and LiDAR), using IMU pre-integration to motion-compensate LiDAR scans between timestamps.
4. **Validation** - the Stage 1 result was independently checked via a dense grid search over tau in [-1, 1]s at 5ms resolution, confirming the optimizer converged to the correct minimum rather than a local one.

## Key Results

- Recovered a consistent inter-sensor time offset of **~73 ms**, confirmed by both Powell optimization (0.293 s convergence time) and independent grid-search validation
- Extended to a joint **7-DOF optimization** over temporal offset and full extrinsic calibration
- Alignment scoring validated qualitatively via LiDAR-to-image projection overlays before and after calibration

## Details About the Files
- **`Camera_Lidar_Temporal_Calibration.ipynb`** - main joint optimization pipeline: IMU pre-integration, LiDAR pipeline, camera pipeline, joint optimization code, and all results.
- **`cam_lidar_extrinsics_calibration_gui.py`** - GUI tool used to find initial extrinsics (rotation and translation) between camera and LiDAR.
- **`utils/extract_topics.py`** - extracts a 5-second data window (t = 10s to t = 15s) from the full recording.
- **`Individual_Pipelines/Camera_Pipeline.ipynb`** - standalone camera pipeline (edge detection, distance transform, scoring).
- **`Individual_Pipelines/imu_utils.py`** - standalone IMU pre-integration pipeline.
- **`Individual_Pipelines/lidar.py`** - standalone LiDAR processing pipeline.

## Data

Collected using the `Huntington.mcap` dataset on the Hunter robot. Dataset and a LiDAR pipeline demo video are available here:
[SharePoint Link](https://northeastern-my.sharepoint.com/:f:/g/personal/lnu_arya_northeastern_edu/Evg01v-PZ55OlbbXd4s0plABuoYSmBLUiZFQTHmBUVXfzg?e=HdgJds)

All other results are in `Camera_Lidar_Temporal_Calibration.ipynb`.

## Contributors

This was a team project. Contributions spanned the joint optimization/temporal calibration pipeline, LiDAR processing, and IMU pre-integration across the group.
