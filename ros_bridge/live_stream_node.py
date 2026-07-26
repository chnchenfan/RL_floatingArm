#!/usr/bin/env python3
"""Receive live Isaac policy ticks over localhost UDP and publish them to ROS."""

from __future__ import annotations

import json
import math
import socket
import sys
import time

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


class LiveStreamNode(Node):
    def __init__(self):
        super().__init__("policy_live_stream")
        host = str(self.declare_parameter("udp_host", "127.0.0.1").value)
        port = int(self.declare_parameter("udp_port", 45831).value)
        self.world_frame = str(
            self.declare_parameter("world_frame", "world").value)
        self.base_frame = str(
            self.declare_parameter("base_frame", "base_link").value)
        self.max_trail_points = int(
            self.declare_parameter("max_trail_points", 1200).value)
        if not 1 <= port <= 65535:
            raise ValueError("udp_port must be in [1, 65535]")
        if self.max_trail_points < 2:
            raise ValueError("max_trail_points must be >= 2")

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.setblocking(False)

        self.joint_publisher = self.create_publisher(
            JointState, "/joint_states", 10)
        # Final RL joint command, republished with the MPC visualizer contract
        # (frame_id 'moving_base_track', names joint1..7, real publish stamp)
        # so trajectory_visualizer_demo.py can be reused unchanged.
        self.command_publisher = self.create_publisher(
            JointState, "/student/joint_command", 10)
        self.actual_publisher = self.create_publisher(
            Marker, "/rollout_ee_actual", 4)
        self.target_publisher = self.create_publisher(
            Marker, "/rollout_ee_target", 4)
        self.transform_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.002, self._poll)

        self.actual_trail = []
        self.target_trail = []
        self.last_stroke = None
        self.last_cycle = None
        self.last_seq = -1
        self.first_packet = True
        self.started_at = time.monotonic()
        self.last_packet_at = None
        self.wait_message_printed = False
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.get_logger().info(
            f"waiting for live Isaac ticks on udp://{host}:{port}")

    def _poll(self):
        received = 0
        while received < 32:
            try:
                payload, _peer = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            received += 1
            try:
                packet = json.loads(payload)
                self._publish(packet)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.get_logger().warning(f"dropped invalid live packet: {exc}")

        now = time.monotonic()
        if (
            self.last_packet_at is None
            and not self.wait_message_printed
            and now - self.started_at > 10.0
        ):
            self.get_logger().info(
                "still waiting; initial Isaac/GPU startup can take a while")
            self.wait_message_printed = True
        if self.last_packet_at is not None and now - self.last_packet_at > 2.0:
            self.get_logger().warning(
                "live Isaac stream has been silent for more than 2 seconds")
            self.last_packet_at = now

    def _publish(self, packet):
        version = int(packet["version"])
        if version not in (1, 2):
            raise ValueError(f"unsupported packet version {packet['version']}")
        sequence = int(packet["seq"])
        if sequence == 0 and self.last_seq > 0:
            self.last_seq = -1
            self.last_cycle = None
        if sequence <= self.last_seq:
            return

        q = _finite_vector(packet, "q", 7)
        dq = _finite_vector(packet, "dq", 7)
        base_position = _finite_vector(packet, "base_pos", 3)
        base_quaternion = _finite_vector(packet, "base_quat", 4)
        ee = _finite_vector(packet, "ee", 3)
        target = _finite_vector(packet, "target", 3)
        cycle = int(packet["cycle"])
        pen_down = bool(packet.get("pen_down", True))
        stroke_id = int(packet.get("stroke_id", 0))
        now = self.get_clock().now().to_msg()

        if cycle != self.last_cycle:
            self.actual_trail.clear()
            self.target_trail.clear()
            self.last_cycle = cycle
        # break line strips on pen-up and on stroke changes so letters and
        # stations are never visually connected
        if not pen_down:
            marker_break = None
            if not self.actual_trail or self.actual_trail[-1] is not None:
                self.actual_trail.append(marker_break)
                self.target_trail.append(marker_break)
            self.last_stroke = None
        else:
            if self.last_stroke is not None and stroke_id != self.last_stroke:
                self.actual_trail.append(None)
                self.target_trail.append(None)
            self.actual_trail.append(ee)
            self.target_trail.append(target)
            self.last_stroke = stroke_id
        if len(self.actual_trail) > self.max_trail_points:
            self.actual_trail.pop(0)
            self.target_trail.pop(0)

        joint_state = JointState()
        joint_state.header.stamp = now
        joint_state.name = self.joint_names
        joint_state.position = q
        joint_state.velocity = dq
        self.joint_publisher.publish(joint_state)

        if "q_cmd" in packet:
            q_cmd = _finite_vector(packet, "q_cmd", 7)
            dq_cmd = (_finite_vector(packet, "dq_cmd", 7)
                      if "dq_cmd" in packet else [0.0] * 7)
            command = JointState()
            command.header.stamp = now
            command.header.frame_id = "moving_base_track"
            command.name = self.joint_names
            command.position = q_cmd
            command.velocity = dq_cmd
            self.command_publisher.publish(command)

        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = base_position[0]
        transform.transform.translation.y = base_position[1]
        transform.transform.translation.z = base_position[2]
        transform.transform.rotation.x = base_quaternion[0]
        transform.transform.rotation.y = base_quaternion[1]
        transform.transform.rotation.z = base_quaternion[2]
        transform.transform.rotation.w = base_quaternion[3]
        self.transform_broadcaster.sendTransform(transform)

        self._publish_trail(
            self.actual_publisher,
            self.actual_trail,
            (0.9, 0.1, 0.1),
            0,
            now,
        )
        self._publish_trail(
            self.target_publisher,
            self.target_trail,
            (0.1, 0.9, 0.1),
            1,
            now,
        )
        if self.first_packet:
            self.get_logger().info(
                "receiving online policy "
                f"checkpoint={packet.get('checkpoint', '?')} "
                f"condition={packet.get('condition', '?')}")
            self.first_packet = False
        self.last_seq = sequence
        self.last_packet_at = time.monotonic()

    def _publish_trail(self, publisher, points, rgb, marker_id, stamp):
        # LINE_LIST of per-segment point pairs; None entries are pen-up or
        # stroke-change separators and never produce a connecting line.
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = stamp
        marker.ns = "rollout"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        previous = None
        for point in points:
            if point is None:
                previous = None
                continue
            if previous is not None:
                marker.points.append(
                    Point(x=previous[0], y=previous[1], z=previous[2]))
                marker.points.append(
                    Point(x=point[0], y=point[1], z=point[2]))
            previous = point
        publisher.publish(marker)

    def destroy_node(self):
        self.socket.close()
        super().destroy_node()


def _finite_vector(packet, name, expected_length):
    values = [float(value) for value in packet[name]]
    if len(values) != expected_length:
        raise ValueError(
            f"{name} has length {len(values)}, expected {expected_length}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} contains a non-finite value")
    return values


def main():
    rclpy.init()
    node = None
    try:
        node = LiveStreamNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        # rclpy's SIGTERM handler invalidates the context while an executor is
        # waiting. That exception is the normal shell-script shutdown path.
        if rclpy.ok():
            print(f"[policy_live_stream] fatal: {exc}", file=sys.stderr)
            raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
