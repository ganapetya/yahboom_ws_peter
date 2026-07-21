#!/usr/bin/env python3
"""Publish a single static JPEG on /camera/color/image_raw, repeatedly, for a
fixed duration -- lets you smoke-test cat_detector (or any image subscriber)
WITHOUT the real Astra camera running. Useful on the bench, or to replay a
known photo (e.g. a labeled dataset image) through the live node to confirm
detection/classification end to end, the same way phase5-status.md §11
Session 1 first validated 5a against a real cat photo before ever touching
the live camera.

Why not just `rclpy.spin(node)` with a timer, like detector_node.py does?
detector_node.py's timer calls back into the SAME node that's spinning, so
it's fine. Here the *natural* thing to try is a timer callback that calls
rclpy.shutdown() once the duration elapses -- don't do that: shutdown() from
inside a callback that spin() is currently executing is racy on this rclpy
version and can hang forever instead of returning. Using spin_once() in an
explicit while-loop sidesteps that entirely: each loop iteration publishes,
then hands control to the executor for up to 0.05s to process any pending
work (there isn't any here, since we don't subscribe to anything -- but
spin_once is still what lets rclpy service its internal timers/discovery),
then control returns to us and we check the clock ourselves. No callback
ever needs to reach into the executor's own run loop.

Usage:
    python3 publish_test_image.py <path/to/photo.jpg> [duration_sec]

Example -- replay a known-labeled white cat photo through the live detector
and confirm the terminal running cat_detector.launch.py logs
"Cat sighting started (white)":
    python3 publish_test_image.py ~/cats/white/cat_20260718_064609_104.jpg 5
"""
import sys
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TestImagePublisher(Node):
    def __init__(self, path):
        super().__init__("test_image_publisher")
        # Plain reliable QoS (the create_publisher default) is fine here even
        # though cat_detector subscribes with qos_profile_sensor_data
        # (best-effort): a RELIABLE publisher can always talk to a
        # BEST_EFFORT subscriber, ROS 2's QoS compatibility rules only block
        # the opposite direction.
        self.pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.bridge = CvBridge()
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"could not read {path}")
        # Convert once outside the loop -- re-publishing the same Image
        # message repeatedly is exactly what a static test image should do;
        # only the header timestamp needs refreshing per publish.
        self.msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

    def tick(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

    rclpy.init()
    node = TestImagePublisher(path)
    end_time = time.monotonic() + duration
    try:
        while time.monotonic() < end_time:
            node.tick()
            rclpy.spin_once(node, timeout_sec=0.05)  # ~15-20 Hz publish rate
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
