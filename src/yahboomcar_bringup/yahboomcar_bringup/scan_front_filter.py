#!/usr/bin/env python3
"""Drop laser ranges outside a trusted forward arc (default ±90°).

The X3 stack mounts the RPLidar low on the chassis; upper structure blocks
the rear hemisphere. Feeding those hits to SLAM creates phantom walls and a
chaotic map. This node publishes /scan_filtered with rear ranges cleared.
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
        self.declare_parameter('trusted_angle_min', -math.pi / 2.0)
        self.declare_parameter('trusted_angle_max', math.pi / 2.0)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.trusted_min = float(self.get_parameter('trusted_angle_min').value)
        self.trusted_max = float(self.get_parameter('trusted_angle_max').value)

        self.pub = self.create_publisher(LaserScan, output_topic, 10)
        self.sub = self.create_subscription(LaserScan, input_topic, self.callback, 10)
        self.get_logger().info(
            f'Keeping scan angles [{math.degrees(self.trusted_min):.0f}, '
            f'{math.degrees(self.trusted_max):.0f}] deg -> {output_topic}'
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
            angle = _normalize_angle(msg.angle_min + i * msg.angle_increment)
            if angle < self.trusted_min or angle > self.trusted_max:
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
