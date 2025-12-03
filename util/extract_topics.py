#!/usr/bin/env python3

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from typing import List, Dict, Optional

import numpy as np
import cv2
from tqdm import tqdm

class McapTimeWindowExtractor:
    def __init__(
        self,
        mcap_path: str,
        camera_topic: str,
        lidar_topic: str,
        imu_topic: str,
        start_time: Optional[float] = None,  # seconds from start of bag
        duration: Optional[float] = None      # duration in seconds
    ):
        self.mcap_path = mcap_path
        self.camera_topic = camera_topic
        self.lidar_topic = lidar_topic
        self.imu_topic = imu_topic
        self.start_time = start_time
        self.duration = duration
        
        # Sensor data storage
        self.camera_frames: List[Dict] = []
        self.lidar_scans: List[Dict] = []
        self.imu_measurements: List[Dict] = []
        
        # Time window bounds
        self.bag_start_time = None
        self.window_start_ns = None
        self.window_end_ns = None
        
        self._topic_handlers = {
            self.camera_topic: self._process_camera,
            self.lidar_topic: self._process_lidar,
            self.imu_topic: self._process_imu,
        }

    def extract(self) -> None:
        print(f"Reading MCAP: {self.mcap_path}")
        print(f"Camera topic: {self.camera_topic}")
        print(f"LiDAR topic:  {self.lidar_topic}")
        print(f"IMU topic:    {self.imu_topic}")
        
        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            
            # Get bag time range
            summary = reader.get_summary()
            if summary and summary.statistics:
                total_msgs = summary.statistics.message_count
                
                # Get first message time to establish reference
                if self.start_time is not None or self.duration is not None:
                    # Find the actual start time of the bag
                    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                        self.bag_start_time = message.log_time
                        break
                    
                    # Calculate window bounds
                    if self.start_time is not None:
                        self.window_start_ns = self.bag_start_time + int(self.start_time * 1e9)
                    else:
                        self.window_start_ns = self.bag_start_time
                    
                    if self.duration is not None:
                        self.window_end_ns = self.window_start_ns + int(self.duration * 1e9)
                    else:
                        self.window_end_ns = float('inf')
                    
                    print()
                    print(f"Extracting time window:")
                    print(f"  Start: +{self.start_time}s from bag start" if self.start_time else "  Start: from beginning")
                    print(f"  Duration: {self.duration}s" if self.duration else "  Duration: until end")
            else:
                total_msgs = None
            
            # Reset reader to start
            with open(self.mcap_path, "rb") as f2:
                reader2 = make_reader(f2, decoder_factories=[DecoderFactory()])
                
                messages_in_window = 0
                with tqdm(desc="Extracting", unit="msg") as pbar:
                    for schema, channel, message, ros_msg in reader2.iter_decoded_messages():
                        topic = channel.topic
                        timestamp_ns = message.log_time
                        
                        # Skip if outside time window
                        if self.window_start_ns is not None and timestamp_ns < self.window_start_ns:
                            continue
                        if self.window_end_ns is not None and timestamp_ns > self.window_end_ns:
                            break
                        
                        # Process message if it's a topic we care about
                        handler = self._topic_handlers.get(topic)
                        if handler is None:
                            continue
                        
                        timestamp_sec = timestamp_ns * 1e-9
                        handler(ros_msg, timestamp_sec, timestamp_ns)
                        
                        messages_in_window += 1
                        pbar.update(1)
        
        print()
        print(f"  Extraction complete:")
        print(f"  Camera frames: {len(self.camera_frames)}")
        print(f"  LiDAR scans:   {len(self.lidar_scans)}")
        print(f"  IMU samples:   {len(self.imu_measurements)}")
        
        if self.camera_frames:
            time_span = self.camera_frames[-1]['timestamp_sec'] - self.camera_frames[0]['timestamp_sec']
            print(f"  Actual time span: {time_span:.2f}s")

    def _process_camera(self, msg, timestamp_sec: float, timestamp_ns: int) -> None:
        if hasattr(msg, 'format'):
            cv_image = self._decode_compressed_image(msg)
        else:
            cv_image = self._decode_ros_image(msg)
        
        height, width = cv_image.shape[:2]
        
        self.camera_frames.append({
            "timestamp_sec": timestamp_sec,
            "timestamp_ns": timestamp_ns,
            "frame_id": msg.header.frame_id,
            "width": width,
            "height": height,
            "image": cv_image,
        })

    def _process_lidar(self, msg, timestamp_sec: float, timestamp_ns: int) -> None:
        point_cloud = self._decode_pointcloud2(msg)
        
        self.lidar_scans.append({
            "timestamp_sec": timestamp_sec,
            "timestamp_ns": timestamp_ns,
            "frame_id": msg.header.frame_id,
            "num_points": point_cloud.shape[0],
            "points": point_cloud,
        })

    def _process_imu(self, msg, timestamp_sec: float, timestamp_ns: int) -> None:
        linear_accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=np.float32)
        
        angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=np.float32)
        
        self.imu_measurements.append({
            "timestamp_sec": timestamp_sec,
            "timestamp_ns": timestamp_ns,
            "frame_id": msg.header.frame_id,
            "linear_acceleration": linear_accel,
            "angular_velocity": angular_vel,
        })

    @staticmethod
    def _decode_compressed_image(compressed_msg) -> np.ndarray:
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _decode_ros_image(image_msg) -> np.ndarray:
        pixel_data = np.frombuffer(image_msg.data, dtype=np.uint8)
        
        if image_msg.encoding == "rgb8":
            image = pixel_data.reshape(image_msg.height, image_msg.width, 3)
            return cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        elif image_msg.encoding == "bgr8":
            return pixel_data.reshape(image_msg.height, image_msg.width, 3).copy()
        elif image_msg.encoding == "mono8":
            return pixel_data.reshape(image_msg.height, image_msg.width).copy()
        else:
            raise ValueError(f"Unsupported encoding: {image_msg.encoding}")

    @staticmethod
    def _decode_pointcloud2(cloud_msg) -> np.ndarray:
        """
        Properly decode PointCloud2 by respecting point_step and field offsets
        """
        # Find X, Y, Z field offsets
        field_map = {}
        for field in cloud_msg.fields:
            if field.name in ['x', 'y', 'z']:
                field_map[field.name] = field.offset
        
        if len(field_map) != 3:
            raise ValueError(f"PointCloud2 missing XYZ fields. Found: {[f.name for f in cloud_msg.fields]}")
        
        # Get point step (bytes per point)
        point_step = cloud_msg.point_step
        num_points = cloud_msg.width * cloud_msg.height
        
        # Ensure we have enough data
        expected_size = num_points * point_step
        actual_size = len(cloud_msg.data)
        
        if actual_size < expected_size:
            num_points = actual_size // point_step
        
        # Create output array
        points = np.zeros((num_points, 3), dtype=np.float32)
        
        # Convert data to numpy array
        data_array = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        
        # Extract each coordinate at its proper offset
        for i, coord_name in enumerate(['x', 'y', 'z']):
            offset = field_map[coord_name]
            
            # Create indices for all points at this coordinate's offset
            indices = np.arange(num_points) * point_step + offset
            
            # Extract float32 values
            for pt_idx, byte_idx in enumerate(indices):
                if byte_idx + 4 <= len(data_array):
                    # Read 4 bytes as float32
                    points[pt_idx, i] = np.frombuffer(
                        data_array[byte_idx:byte_idx+4].tobytes(), 
                        dtype=np.float32
                    )[0]
        
        # Filter invalid points
        valid_mask = np.all(np.isfinite(points), axis=1)
        valid_points = points[valid_mask]
        
        # Remove exact zeros (common invalid marker)
        non_zero_mask = ~np.all(valid_points == 0, axis=1)
        
        return valid_points[non_zero_mask]

# if __name__ == "__main__":
#     # Example 1: Extract 5 seconds starting from 10 seconds into the bag
#     # You would need to manually visualize what will be your data start 
#     extractor = McapTimeWindowExtractor(
#         mcap_path="dataset/bag2_forsyth_street_all.mcap",
#         camera_topic="/cam_sync/cam0/image_raw/compressed",
#         lidar_topic="/ouster/points",
#         imu_topic="/vectornav/imu_uncompensated",
#         start_time=10.0,    # Start at 10 seconds
#         duration=5.0        # Extract 5 seconds worth
#     )
    
#     extractor.extract()
    
#     if extractor.camera_frames:
#         print(f"\n[CAM]")
#         print(f"  First frame t: {extractor.camera_frames[0]['timestamp_sec']:.3f}s")
#         print(f"  Last frame t:  {extractor.camera_frames[-1]['timestamp_sec']:.3f}s")
#         print(f"  Image shape: {extractor.camera_frames[0]['image'].shape}")
#         print(f"  Total frames: {len(extractor.camera_frames)}")
    
#     if extractor.lidar_scans:
#         print(f"\n[LIDAR]")
#         print(f"  First scan t: {extractor.lidar_scans[0]['timestamp_sec']:.3f}s")
#         print(f"  Last scan t:  {extractor.lidar_scans[-1]['timestamp_sec']:.3f}s")
#         print(f"  Points shape: {extractor.lidar_scans[0]['points'].shape}")
#         print(f"  Total scans: {len(extractor.lidar_scans)}")
    
#     if extractor.imu_measurements:
#         print(f"\n[IMU]")
#         print(f"  First meas t: {extractor.imu_measurements[0]['timestamp_sec']:.3f}s")
#         print(f"  Last meas t:  {extractor.imu_measurements[-1]['timestamp_sec']:.3f}s")
#         print(f"  Total measurements: {len(extractor.imu_measurements)}")