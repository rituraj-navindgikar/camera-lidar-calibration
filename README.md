# Camera-Lidar-Temporal-Calibration By Group 1

### Details About The Files
- <b>Camera_Lidar_Temporal_Calibration.ipynb -</b> This is the main joint optimization pipeline used for the temporal calibration. This file includes the imu preintegration pipeline, lidar pipeline, the camera pipeline, the joint optimization code, and all the results.
- <b>cam_lidar_extrinsics_calibration_gui.py -</b> This file was used to find the extrinsics (Rotation and Translation) between the camera and the lidar.
- <b>utils/extract_topics.py -</b> This file was used to extract data for a duration of 5 seconds (from t = 10 seconds to t = 15 seconds).
- <b>Individual_Pipelines/Camera_Pipeline.ipynb -</b> This contains code for the camera pipeline.
- <b>Individual_Pipelines/imu_utils.py -</b> This contains code for the imu preintegration pipeline. 
- <b>Individual_Pipelines/lidar.py -</b> This contains the code for the lidar pipeline. 

### We referred to the "Temporal and Spatial Online Integrated Calibration for Camera and LiDAR" paper.
<b>Link:</b> https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9921858

### We used the Huntington.mcap dataset collected using the Hunter Robot.

### Please find the dataset and one of the lidar pipeline videos in the drive folder below. 

### All other results are in the ``Camera_Lidar_Temporal_Calibration.ipynb`` notebook.

<b>Link:</b> https://northeastern-my.sharepoint.com/:f:/g/personal/lnu_arya_northeastern_edu/Evg01v-PZ55OlbbXd4s0plABuoYSmBLUiZFQTHmBUVXfzg?e=HdgJds







