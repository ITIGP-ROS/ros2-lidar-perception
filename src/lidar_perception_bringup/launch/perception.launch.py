"""Bring up the LiDAR perception pipeline.

Mirrors the old lidar_object_detect_bringup layout: a detector node plus the
visualiser. The difference is that detection AND tracking now run in the SAME
process -- the TensorRT detector hands boxes straight to the C++ AB3DMOT
tracker, and the node publishes object_detection_msgs/Object3dArray with track
ids already filled in. There is no separate tracker node to launch.

Everything is configured from params/<profile>.yaml, including the model path
and the visualiser's display threshold, so the common case is just:

    ros2 launch lidar_perception_bringup perception.launch.py

Any of the launch arguments below override the file when given; leaving one
unset keeps whatever the file says (an empty override is NOT pushed onto the
node, which is why this uses OpaqueFunction rather than a plain parameter dict).

    ros2 launch lidar_perception_bringup perception.launch.py \
        profile:=livox model_path:=/abs/path/to/model.onnx

See docs/RUNNING.md for the full manual test procedure.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = [
    ('profile', 'kitti',
     "Parameter profile: 'kitti' (bag replay) or 'livox' (vehicle)."),
    ('model_path', '',
     'Absolute path to the PointPillars .onnx. Overrides the profile file.'),
    ('input_topic', '',
     'Point cloud topic. Overrides the profile file.'),
    ('score_threshold', '',
     'Runtime cut-off on what the detector PUBLISHES. Overrides the profile '
     'file. Can only raise the compiled SCORE_THRESH, never lower it.'),
    ('confidence_threshold', '',
     'Visualiser-only display cut-off. Overrides the profile file. Does NOT '
     'affect detection; that threshold is SCORE_THRESH in config/*.yaml.'),
    ('visualize', 'true',
     'Also run the RViz marker visualiser.'),
]


def launch_setup(context, *_, **__):
    share = get_package_share_directory('lidar_perception_bringup')
    val = {name: LaunchConfiguration(name).perform(context) for name, _, _ in ARGS}

    params_file = os.path.join(share, 'params', val['profile'] + '.yaml')
    if not os.path.isfile(params_file):
        raise RuntimeError(
            f"no such profile '{val['profile']}': {params_file} does not exist")

    # Only push overrides that were actually given. Passing an empty string
    # through would blank out the value the profile file supplies.
    detector_overrides = {}
    if val['model_path']:
        detector_overrides['model_path'] = val['model_path']
    if val['input_topic']:
        detector_overrides['input_topic'] = val['input_topic']
    if val['score_threshold']:
        detector_overrides['score_threshold'] = float(val['score_threshold'])

    viz_overrides = {}
    if val['confidence_threshold']:
        viz_overrides['confidence_threshold'] = float(val['confidence_threshold'])

    nodes = [Node(
        package='cuda_pointpillars_ros',
        executable='pc_process',
        name='cuda_pointpillars_node',
        output='screen',
        parameters=[params_file] + ([detector_overrides] if detector_overrides else []),
    )]
    if val['visualize'].lower() in ('true', '1', 'yes'):
        nodes.append(Node(
            package='object_visualization',
            executable='object3d_visualizer_node',
            name='object3d_visualizer_node',
            output='screen',
            parameters=[params_file] + ([viz_overrides] if viz_overrides else []),
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
        + [OpaqueFunction(function=launch_setup)])
