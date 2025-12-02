import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
import time
from extract_topics import McapTimeWindowExtractor
from imu_utils import IMUPreintegrator

## Data Paths & Topics
MCAP_PATH = "bag2_forsyth_street_all.mcap"
LIDAR_TOPIC = "/ouster/points"
CAMERA_TOPIC = "/cam_sync/cam0/image_raw/compressed"
IMU_TOPIC = "/vectornav/imu_uncompensated"

# LiDAR Parameters
H, W = 128, 1024 # LiDAR Resolution
FOV_UP, FOV_DOWN = 22.5, -22.5 # Vertical FOV

## ROS2 Node for Visualization in RViz
class VisualizationNode(Node):
    def __init__(self):
        super().__init__('imu_full_vis')
        
        # TOPIC 1: Raw Single Frame (Red) - The baseline 3D Map
        self.pub_raw_single = self.create_publisher(PointCloud2, '/lidar_raw_single', 10)
        
        # TOPIC 2: Raw Dense Map (Blue) - The Full 3D Map (Stacked & Corrected)
        self.pub_raw_dense = self.create_publisher(PointCloud2, '/lidar_raw_dense', 10)
        
        # TOPIC 3: Corrected Edges (Green) - The Features for Calibration
        self.pub_edges = self.create_publisher(PointCloud2, '/lidar_edges_corrected', 10)
        
        print("ROS2 Node Initialized. Publishing 3 topics.")

    ## ROS2 Publish Function
    def publish_all(self, raw_single, raw_dense, edges):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "velodyne" 

        if raw_single is not None:
            self.pub_raw_single.publish(pc2.create_cloud_xyz32(header, raw_single))
        if raw_dense is not None:
            self.pub_raw_dense.publish(pc2.create_cloud_xyz32(header, raw_dense))
        if edges is not None:
            self.pub_edges.publish(pc2.create_cloud_xyz32(header, edges))

## Projecjection of LiDAR points to LiDAR's Dimensions
def project_to_range_image(points, height=128, width=1024):
    """ Robust Grid Restoration (Fixes Divide-by-Zero) """
    total = height * width
    if points.shape[0] == total: return points.reshape(height, width, 3)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    
    # Mask valid points to prevent crash
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

## Extract Edges from the LiDAR Map
def extract_edges(grid_points):
    """ Extract Edges > 1.0m jump """
    ranges = np.linalg.norm(grid_points, axis=2)
    diff = np.abs(np.diff(ranges, axis=1))
    diff = np.pad(diff, ((0,0), (0,1)), constant_values=0)
    return grid_points[diff > 1.0]

## Apply 4x4 Transformation to Point Cloud (Initialize Homogeneous Coordinates)
def apply_transform(points, T):
    """ Applies 4x4 Transform to Nx3 points """
    if len(points) == 0: return points
    ones = np.ones((len(points), 1))
    pts_hom = np.hstack((points, ones))
    return (T @ pts_hom.T).T[:, :3]

def main():
    rclpy.init()
    node = VisualizationNode()

    print(f"Loading MCAP: {MCAP_PATH} ...")
    extractor = McapTimeWindowExtractor(
        mcap_path=MCAP_PATH, camera_topic=CAMERA_TOPIC,
        lidar_topic=LIDAR_TOPIC, imu_topic=IMU_TOPIC,
        start_time=0.0, duration=None 
    )
    extractor.extract()
    print("Starting Playback...")

    imu_integrator = IMUPreintegrator(gravity=np.array([0, 0, 9.81]))
    
    # Window stores: {'t': timestamp, 'edges': points, 'raw': points}
    window = []

    try:
        for i, scan in enumerate(extractor.lidar_scans):
            if not rclpy.ok(): break
            
            raw_points = scan['points']
            timestamp = scan['timestamp_sec']
            
            # 1. Feature Extraction
            grid = project_to_range_image(raw_points, H, W)
            edges = extract_edges(grid)
            
            # 2. Buffer
            window.append({'t': timestamp, 'edges': edges, 'raw': raw_points})
            if len(window) > 3: window.pop(0)

            if len(window) == 3:
                f_prev, f_curr, f_next = window[0], window[1], window[2]
                
                ## IMU Motion Compensation
                # 1. Previous to Current scan
                T_p2c = imu_integrator.preintegrate(extractor.imu_measurements, f_prev['t'], f_curr['t'])
                edges_prev = apply_transform(f_prev['edges'], T_p2c)
                raw_prev = apply_transform(f_prev['raw'], T_p2c)

                # 2. Next to Current scan
                T_c2n = imu_integrator.preintegrate(extractor.imu_measurements, f_curr['t'], f_next['t'])
                T_n2c = np.linalg.inv(T_c2n)
                edges_next = apply_transform(f_next['edges'], T_n2c)
                raw_next = apply_transform(f_next['raw'], T_n2c)
                
                # 3. Stack (Densification)
                dense_edges_corrected = np.vstack([edges_prev, f_curr['edges'], edges_next])
                dense_raw_corrected = np.vstack([raw_prev, f_curr['raw'], raw_next])
                
                # 4. Publish
                node.publish_all(f_curr['raw'], dense_raw_corrected, dense_edges_corrected)
                
                print(f"Scan {i}: Densified Edge Count: {len(dense_edges_corrected)}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()