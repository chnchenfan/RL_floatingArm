"""View an Isaac rollout in RViz.

Runs robot_state_publisher(arm.urdf) + the rollout replay node + (optionally)
rviz2. Reads the URDF from file and passes it as robot_description so ROS arg
parsing does not choke on multi-line XML (a plain `ros2 run` CLI does).

    source /opt/ros/humble/setup.bash
    source ~/code/windylab_ws/install/setup.bash      # for package:// meshes
    ros2 launch ros_bridge/view_rollout.launch.py \
        rollout:=data/rollout_synth.csv rate:=50.0 use_rviz:=true

Add the tracking plot in another terminal:
    ros2 run manipulator trajectory_visualizer_demo.py --ros-args \
        -p trajectory_mode:=circle -p frame_id:=base_link
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WS = "/home/windylab/code/windylab_ws"
URDF = os.path.join(WS, "src/arm-platform/config/arm.urdf")
BRIDGE = os.path.dirname(os.path.abspath(__file__))
RVIZ = os.path.join(BRIDGE, "arm_rl.rviz")   # preconfigured: world + robot + trails


def generate_launch_description():
    rollout = LaunchConfiguration("rollout")
    rate = LaunchConfiguration("rate")
    use_rviz = LaunchConfiguration("use_rviz")

    with open(URDF, "r") as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            "rollout",
            default_value="/home/windylab/code/isaac_arm_rl/data/rollout_task.csv"),
        DeclareLaunchArgument("rate", default_value="50.0"),
        DeclareLaunchArgument("use_rviz", default_value="true"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        ExecuteProcess(
            cmd=["python3", os.path.join(BRIDGE, "replay_rollout_node.py"),
                 "--ros-args",
                 "-p", ["rollout_file:=", rollout],
                 "-p", ["publish_rate_hz:=", rate],
                 "-p", "loop:=true"],
            output="screen",
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=["-d", RVIZ],
            condition=IfCondition(use_rviz),
        ),
    ])
