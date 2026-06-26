#!/usr/bin/env python3
"""Keep a trusted angular arc of the laser scan; clear everything else.

The arc is defined by a center bearing and a half-width (radians). A range whose
bearing is within +/- half-width of the center is kept; all others are set to +inf.

X3 NOTE: the RPLiDAR A1 is mounted ~180 deg rotated relative to the URDF, so in the
RAW scan frame angle 0 points to the robot's REAR (the antennas/wires) and angle
+/-pi points to the robot's FRONT. To trust the clean front, launch with
trusted_center = pi (see myscripts2/t2.5.sh). The companion URDF fix adds 180 deg
yaw to base_link->laser_link so the kept points are placed in the correct direction.
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class ScanFrontFilter(Node):
    def __init__(self) -> None:
        super().__init__('scan_front_filter')
        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_filtered')
        # Bearing (rad) the trusted arc is centered on, and its half-width (rad).
        # Default: center 0 (forward), half-width 90 deg. Override per robot.
        self.declare_parameter('trusted_center', 0.0)
        self.declare_parameter('trusted_halfwidth', math.pi / 2.0)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.center = float(self.get_parameter('trusted_center').value)
        self.halfwidth = abs(float(self.get_parameter('trusted_halfwidth').value))

        self.pub = self.create_publisher(LaserScan, output_topic, 10)
        self.sub = self.create_subscription(LaserScan, input_topic, self.callback, 10)
        self.get_logger().info(
            f'Trusting arc center={math.degrees(self.center):.0f} deg '
            f'+/-{math.degrees(self.halfwidth):.0f} deg -> {output_topic}'
        )

    def callback(self, msg: LaserScan) -> None:
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = list(msg.ranges)
        out.intensities = list(msg.intensities)

        for i, _ in enumerate(out.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if abs(_normalize_angle(angle - self.center)) > self.halfwidth:
                out.ranges[i] = float('inf')
                if i < len(out.intensities):
                    out.intensities[i] = 0.0

        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = ScanFrontFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
