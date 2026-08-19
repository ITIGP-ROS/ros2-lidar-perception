import rclpy
from rclpy.node import Node
import ros2_numpy as rnp
from sensor_msgs.msg import PointCloud2
import torch
from .model import PointPillars
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np
from .utils import bbox3d2corners
from .tracking import Track3D
from object_detection_msgs.msg import Object3d, Object3dArray
from geometry_msgs.msg import Point

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy
)

CLASSES = {
    'Pedestrian': 0, 
    'Cyclist': 1, 
    'Car': 2        
}


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 



def to_kitti_format(pc):
    pc = np.asarray(pc)

    # Case 1: structured PointCloud2 (64x1024 or similar)
    if pc.ndim == 2 and pc.dtype.names is not None:
        x = pc['x'].reshape(-1)
        y = pc['y'].reshape(-1)
        z = pc['z'].reshape(-1)

        if 'intensity' in pc.dtype.names:
            i = pc['intensity'].reshape(-1)
        else:
            i = np.zeros_like(x)

        return np.stack([x, y, z, i], axis=-1).astype(np.float32)

    # Case 2: already flat structured array
    if pc.ndim == 1 and pc.dtype.names is not None:
        x = pc['x']
        y = pc['y']
        z = pc['z']
        i = pc['intensity'] if 'intensity' in pc.dtype.names else np.zeros_like(x)

        return np.stack([x, y, z, i], axis=-1).astype(np.float32)

    # Case 3: already Nx4 float array
    if isinstance(pc, np.ndarray) and pc.ndim == 2 and pc.shape[1] >= 3:
        if pc.shape[1] == 3:
            i = np.zeros((pc.shape[0], 1), dtype=np.float32)
            return np.hstack([pc, i]).astype(np.float32)
        return pc[:, :4].astype(np.float32)

    raise ValueError(f"Unknown point cloud format: {type(pc)}, shape={pc.shape}")

class LidarObjectDetectorNode(Node):

    def __init__(self, weights, device=torch.device('cuda')):
        super().__init__('lidar_object_detector_node')

        self.declare_parameter("output_frame_id", "velo_link")
        self.output_frame_id = self.get_parameter("output_frame_id").get_parameter_value().string_value
        
        self.lidar_subscription = self.create_subscription(
            msg_type = PointCloud2,
            topic = 'kitti/velo',
            callback = self.lidar_point_cloud_callback,
            # QOS to be compatible with bag file
            qos_profile = QoSProfile(
                            reliability=QoSReliabilityPolicy.RELIABLE,
                            history=QoSHistoryPolicy.KEEP_LAST,
                            durability=QoSDurabilityPolicy.VOLATILE,
                            depth=1
                        )
        )

        self.detections_publisher = self.create_publisher(Object3dArray, 'object_detections_3d', 10)

        # 3D multi-object tracker (vendored AB3DMOT core) that assigns stable
        # track IDs and predicts boxes during missed detections
        self.tracker = Track3D()

        # setup model
        self.device = device
        self.model = PointPillars(nclasses=len(CLASSES)).to(self.device)
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()        


    def lidar_point_cloud_callback(self, lidar_msg: PointCloud2):
        
        point_cloud_numpy = rnp.numpify(lidar_msg)
        
        ## KITTI format        
        ## Compatible with my Bag file format
        
        # point_cloud_numpy = np.stack([
        #     point_cloud_numpy['x'],
        #     point_cloud_numpy['y'],
        #     point_cloud_numpy['z'],
        #     point_cloud_numpy['intensity']
        # ], axis=1).astype(np.float32)
        
        point_cloud_numpy = to_kitti_format(point_cloud_numpy)
        
        
        point_cloud_numpy = point_range_filter(point_cloud_numpy)
        point_cloud_tensor = torch.from_numpy(point_cloud_numpy).to(self.device)

        # inference
        with torch.no_grad():
            outputs = self.model(batched_pts=[point_cloud_tensor])

        self.get_logger().info(f"model output type: {type(outputs)}")

        # -- Some safety checks to make avoid the crashes while using th Mid-360 bag files--
        
        # ---- SAFE UNPACKING ----
        if isinstance(outputs, (list, tuple)):
            results = outputs[0]
        elif isinstance(outputs, dict):
            results = outputs
        else:
            self.get_logger().error(f"Unexpected model output type: {type(outputs)}")
            return

        # extra safety check: the model returns a plain list when there are no detections
        if not isinstance(results, dict) or 'lidar_bboxes' not in results:
            self.get_logger().info("empty")
            # still advance the tracker so existing tracks age out correctly
            tracks = self.tracker.update(
                np.empty((0, 7), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64)
            )
        else:
            # track detections across frames: stable IDs, Kalman prediction during misses
            tracks = self.tracker.update(
                results['lidar_bboxes'],
                results['scores'],
                results['labels']
            )
        # ------------------------------------------------------------------

        detection_array = Object3dArray()
        for track in tracks:
            # track: [x, y, z, w, l, h, yaw, score, label, track_id]
            bbox = bbox3d2corners(track[:7].reshape(1, 7))[0]
            detection = Object3d()
            detection.label = int(track[8])
            detection.confidence_score = float(track[7])
            detection.track_id = int(track[9])
            for i in range(len(bbox)):
                corner = Point()
                corner.x = float(bbox[i][0])
                corner.y = float(bbox[i][1])
                corner.z = float(bbox[i][2])
                detection.bounding_box.corners[i] = corner
            detection_array.objects.append(detection)

        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = self.output_frame_id
        self.detections_publisher.publish(detection_array)
        self.get_logger().info("Successfully ran inference on lidar scan")


def main(args=None):
    rclpy.init(args=args)

    package_name = 'lidar_object_detection'
    share_dir = os.path.dirname(get_package_share_directory(package_name))
    weights = os.path.join(share_dir, package_name, 'weights/epoch_160.pth')

    lidar_object_detector_node = LidarObjectDetectorNode(weights)

    rclpy.spin(lidar_object_detector_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lidar_object_detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()