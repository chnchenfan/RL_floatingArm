#!/usr/bin/env python3
"""Replay an Isaac/IK rollout onto /joint_states (+ base TF + trail markers) for
RViz and trajectory_visualizer. No changes to the visualizer needed.

Publishes:
  - /joint_states           : the 7 arm joints
  - TF world->base_link     : if the rollout has base pose (base shaking)
  - /rollout_ee_actual      : Marker LINE_STRIP, red   (drawn EE path, world)
  - /rollout_ee_target      : Marker LINE_STRIP, green (target shape, world)

Run (after sourcing ROS 2 + robot_state_publisher/urdf):
    python3 ros_bridge/replay_rollout_node.py --ros-args \
        -p rollout_file:=data/rollout_task.csv -p publish_rate_hz:=50.0 -p loop:=true
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rollout_io import read_rollout  # noqa: E402


class RolloutReplay(Node):
    def __init__(self):
        super().__init__("rollout_replay")
        self.rollout_file = self.declare_parameter("rollout_file", "").value
        self.publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 50.0).value)
        self.loop = bool(self.declare_parameter("loop", True).value)
        self.time_scale = float(self.declare_parameter("time_scale", 1.0).value)
        self.topic = self.declare_parameter("topic", "/joint_states").value
        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.base_frame = self.declare_parameter("base_frame", "base_link").value

        if not self.rollout_file or not os.path.isfile(self.rollout_file):
            raise FileNotFoundError(f"rollout_file not found: {self.rollout_file}")
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be > 0")

        r = read_rollout(self.rollout_file)
        self.t = r["t"]
        self.q = r["q"]
        self.vel = np.vstack([np.zeros((1, 7)), np.diff(self.q, axis=0)
                              / np.maximum(np.diff(self.t)[:, None], 1e-6)])
        self.joint_names = r["joint_names"]
        self.base_pos = r["base_pos"]
        self.base_quat = r["base_quat"]
        self.ee = r["ee"]
        self.tgt = r["tgt"]
        self.drawing = r.get("drawing")
        if self.drawing is None and self.ee is not None:
            self.drawing = np.ones(len(self.ee), dtype=bool)
        self.t0 = float(self.t[0])
        self.duration = float(self.t[-1] - self.t[0])

        self.pub = self.create_publisher(JointState, self.topic, 10)
        self.tf_bc = TransformBroadcaster(self) if self.base_pos is not None else None
        self.m_act = self.create_publisher(Marker, "/rollout_ee_actual", 4)
        self.m_tgt = self.create_publisher(Marker, "/rollout_ee_target", 4)

        self.start_ns = self.get_clock().now().nanoseconds
        self._last_idx = -1
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._tick)
        self.get_logger().info(
            f"replaying {self.rollout_file}: {len(self.t)} samples, "
            f"{self.duration:.2f}s, base={'yes' if self.base_pos is not None else 'no'}, "
            f"rate={self.publish_rate_hz}Hz loop={self.loop}")

    def _tick(self):
        elapsed = (self.get_clock().now().nanoseconds - self.start_ns) * 1e-9
        tau = elapsed * self.time_scale
        wrapped = False
        if tau > self.duration:
            if self.loop:
                self.start_ns = self.get_clock().now().nanoseconds
                tau = 0.0
                wrapped = True
            else:
                tau = self.duration
        idx = int(np.searchsorted(self.t, self.t0 + tau, side="right") - 1)
        idx = max(0, min(idx, len(self.t) - 1))
        now = self.get_clock().now().to_msg()

        msg = JointState()
        msg.header.stamp = now
        msg.name = list(self.joint_names)
        msg.position = [float(v) for v in self.q[idx]]
        msg.velocity = [float(v) for v in self.vel[idx]]
        self.pub.publish(msg)

        if self.tf_bc is not None:
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = self.world_frame
            tf.child_frame_id = self.base_frame
            p = self.base_pos[idx]; qq = self.base_quat[idx]
            tf.transform.translation.x = float(p[0])
            tf.transform.translation.y = float(p[1])
            tf.transform.translation.z = float(p[2])
            tf.transform.rotation.x = float(qq[0])
            tf.transform.rotation.y = float(qq[1])
            tf.transform.rotation.z = float(qq[2])
            tf.transform.rotation.w = float(qq[3])
            self.tf_bc.sendTransform(tf)

        # accumulate EE trails (world frame), ONLY over drawing samples so the
        # four shapes show cleanly (no arcs during the turns); clear on loop wrap
        if wrapped:
            self._last_idx = -1
        if self.ee is not None and idx != self._last_idx:
            mask = self.drawing[:idx + 1]
            self._publish_trail(self.m_act, self.ee[:idx + 1][mask], (0.9, 0.1, 0.1), 0, now)
            self._publish_trail(self.m_tgt, self.tgt[:idx + 1][mask], (0.1, 0.9, 0.1), 1, now)
        self._last_idx = idx

    def _publish_trail(self, pub, pts, rgb, mid, stamp):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = stamp
        m.ns = "rollout"
        m.id = mid
        m.type = Marker.LINE_STRIP if len(pts) > 1 else Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.005
        m.scale.y = 0.005
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        m.pose.orientation.w = 1.0
        step = max(1, len(pts) // 3000)
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    for p in pts[::step]]
        pub.publish(m)


def main():
    rclpy.init()
    try:
        node = RolloutReplay()
    except Exception as e:
        print(f"[rollout_replay] fatal: {e}", file=sys.stderr)
        rclpy.try_shutdown()
        raise
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
