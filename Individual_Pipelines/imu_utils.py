import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Dict

class IMUPreintegrator:
    def __init__(self,
                 gravity: np.ndarray = np.array([0, 0, 9.81]),
                 verbose: bool = False):
        """
        Calculates relative motion (Dead Reckoning) from IMU data.
        
        :param gravity: Gravity vector in the IMU frame. 
                        Usually [0, 0, 9.81] if Z is UP.
                        Change to [0, 0, -9.81] if Z is DOWN.
        """
        self.gravity = gravity
        self.verbose = verbose
        self.reset()
        
    def reset(self):
        self.delta_R = Rotation.identity()
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.delta_t = 0.0
        
    def preintegrate(self, 
                     imu_measurements: List[Dict],
                     t_start: float,
                     t_end: float) -> np.ndarray:
        """
        Integrates IMU messages between t_start and t_end.
        Returns: 4x4 Transformation Matrix (T_v) representing motion.
        """
        
        # 1. Filter IMU data to strict time window
        # We assume the list is sorted by time
        relevant_imu = [
            m for m in imu_measurements 
            if m['timestamp_sec'] >= t_start and m['timestamp_sec'] <= t_end
        ]

        if len(relevant_imu) < 2:
            if self.verbose: 
                print(f"Warning: Not enough IMU data between {t_start:.3f} and {t_end:.3f}")
            return np.eye(4) # Return Identity (No motion)
        
        self.delta_t = t_end - t_start
        self.reset()
        
        # 2. Integration Loop
        prev_time = relevant_imu[0]['timestamp_sec']
        
        for i in range(1, len(relevant_imu)):
            imu = relevant_imu[i]
            t_curr = imu['timestamp_sec']
            dt = t_curr - prev_time
            
            if dt <= 0: continue
            
            # Extract data (Expecting numpy arrays from extract_topics.py)
            acc = imu['linear_acceleration']
            gyro = imu['angular_velocity']
            
            self._integrate_step(acc, gyro, dt)
            prev_time = t_curr
            
        return self.get_transform_matrix()
    
    def _integrate_step(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """ Standard Strapdown Inertial Navigation integration (Euler) """
        
        # 1. Update Rotation (Gyro)
        # Calculate angle change
        delta_angle = gyro * dt
        delta_rotation = Rotation.from_rotvec(delta_angle)
        
        # 2. Update Position & Velocity (Accel)
        # Rotate acceleration from Body Frame to World Frame
        acc_world = self.delta_R.apply(acc)
        
        # Subtract Gravity (Crucial!)
        acc_net = acc_world - self.gravity
        
        # Euler Integration: p = p + v*dt + 0.5*a*dt^2
        self.delta_p += self.delta_v * dt + 0.5 * acc_net * dt**2
        self.delta_v += acc_net * dt
        
        # Update Accumulator
        self.delta_R = self.delta_R * delta_rotation
    
    def get_transform_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.delta_R.as_matrix()
        T[:3, 3] = self.delta_p
        return T