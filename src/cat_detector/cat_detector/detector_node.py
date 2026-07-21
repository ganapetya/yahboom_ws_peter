#!/usr/bin/env python3
import datetime
import glob
import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image as PILImage

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray, Detection2D, \
    ObjectHypothesisWithPose, BoundingBox2D

# Padding fraction must match collect_dataset.py's PAD_FRAC (0.15) -- the
# classifier was trained on crops padded this much, so live crops need the
# same context around the box or accuracy silently degrades.
ID_CROP_PAD_FRAC = 0.15


class CatDetector(Node):
    def __init__(self):
        super().__init__("cat_detector")

        self.model_path = self.declare_parameter("model_path", "yolov8n.pt").value
        self.device = self.declare_parameter("device", "cuda:0").value
        self.imgsz = int(self.declare_parameter("imgsz", 640).value)
        self.target_classes = list(self.declare_parameter("target_classes", [15]).value)
        self.conf = float(self.declare_parameter("conf_threshold", 0.45).value)
        self.iou = float(self.declare_parameter("iou_threshold", 0.50).value)
        image_topic = self.declare_parameter("image_topic", "/camera/color/image_raw").value
        det_topic = self.declare_parameter("detections_topic", "/cat_detector/detections").value
        ann_topic = self.declare_parameter("annotated_topic", "/cat_detector/image_annotated").value
        self.publish_annotated = bool(self.declare_parameter("publish_annotated", True).value)
        self.log_fps_every = float(self.declare_parameter("log_fps_every_sec", 5.0).value)

        # --- Persistent dataset capture on sighting (Phase 5b flywheel) ---
        self.save_on_detection = bool(self.declare_parameter("save_on_detection", True).value)
        self.detection_image_dir = self.declare_parameter(
            "detection_image_dir", "/home/jetson/cat_patrol_data/detections").value
        self.sighting_gap_sec = float(self.declare_parameter("sighting_gap_sec", 3.0).value)
        self.detection_save_interval_sec = float(
            self.declare_parameter("detection_save_interval_sec", 5.0).value)
        self.max_detection_images = int(
            self.declare_parameter("max_detection_images", 500).value)
        os.makedirs(self.detection_image_dir, exist_ok=True)
        self._last_seen_time = None    # monotonic time of the last frame with a cat
        self._last_saved_time = None   # monotonic time of the last saved dataset frame

        # --- Beep on sighting ---
        self.beep_on_detection = bool(self.declare_parameter("beep_on_detection", True).value)
        beep_topic = self.declare_parameter("beep_topic", "Buzzer").value
        self.beep_duration_sec = float(self.declare_parameter("beep_duration_sec", 0.15).value)
        self.beep_pub = (
            self.create_publisher(Bool, beep_topic, 10) if self.beep_on_detection else None
        )

        # --- Per-cat identification (Phase 5b) ---
        self.enable_identification = bool(
            self.declare_parameter("enable_identification", True).value)
        self.classifier_model_path = self.declare_parameter(
            "classifier_model_path",
            "/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/cat_detector/models/cat_id_classifier.pt"
        ).value
        self.classifier = None
        self.classifier_classes = None
        if self.enable_identification:
            self._load_classifier()

        self.get_logger().info(f"Loading model {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        # Latest-wins queue of 1: the callback only STORES the newest frame;
        # a timer does the (slow) inference. If inference is slower than the
        # camera, intermediate frames are simply overwritten -> dropped stale,
        # bounded memory. This is the drop-stale requirement in Python.
        self._latest = None

        self.det_pub = self.create_publisher(Detection2DArray, det_topic, 10)
        self.ann_pub = self.create_publisher(Image, ann_topic, 10) if self.publish_annotated else None
        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)

        # Inference loop runs as fast as it can; overwrite semantics do the throttling.
        self.create_timer(0.001, self._infer)
        self._frames = 0
        self._t0 = time.monotonic()

        self.get_logger().info(
            f"cat_detector started. image_topic={image_topic} "
            f"detections_topic={det_topic} detection_image_dir={self.detection_image_dir}")

    def _load_classifier(self):
        if not os.path.isfile(self.classifier_model_path):
            self.get_logger().warning(
                f"enable_identification=true but no classifier at "
                f"{self.classifier_model_path} -- falling back to generic "
                f"'cat' detection only (run train_classifier.py first).")
            self.enable_identification = False
            return

        checkpoint = torch.load(self.classifier_model_path, map_location=self.device)
        self.classifier_classes = checkpoint["classes"]

        net = models.mobilenet_v2(weights=None)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, len(self.classifier_classes))
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        self.classifier = net.to(self.device)

        self.classifier_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.get_logger().info(
            f"Loaded per-cat classifier from {self.classifier_model_path} "
            f"(classes={self.classifier_classes}, test_acc="
            f"{checkpoint.get('test_acc', 'unknown')})")

    def _classify_crop(self, frame_bgr, box):
        """Crop the detection box (padded to match training, ID_CROP_PAD_FRAC)
        out of the BGR frame and run the per-cat classifier on it. Returns
        (label, confidence) or (None, None) if identification is disabled."""
        if not self.enable_identification or self.classifier is None:
            return None, None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * ID_CROP_PAD_FRAC, bh * ID_CROP_PAD_FRAC
        cx1 = max(0, int(x1 - px))
        cy1 = max(0, int(y1 - py))
        cx2 = min(w, int(x2 + px))
        cy2 = min(h, int(y2 + py))
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None, None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        tensor = self.classifier_tf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.classifier(tensor), dim=1)[0]
        idx = int(probs.argmax())
        return self.classifier_classes[idx], float(probs[idx])

    def _on_image(self, msg):
        self._latest = msg  # overwrite -> latest-wins

    def _infer(self):
        msg = self._latest
        if msg is None:
            return
        self._latest = None  # consume

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            classes=self.target_classes, device=self.device, verbose=False)
        r = results[0]

        out = Detection2DArray()
        out.header = msg.header
        identities = []  # (label, conf) or (None, None) per box, for annotation
        for b in r.boxes:
            cls_id = int(b.cls[0])
            score = float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            d = Detection2D()
            d.header = msg.header
            bb = BoundingBox2D()
            bb.center.position.x = (x1 + x2) / 2.0
            bb.center.position.y = (y1 + y2) / 2.0
            bb.size_x = float(x2 - x1)
            bb.size_y = float(y2 - y1)
            d.bbox = bb
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(cls_id)
            hyp.hypothesis.score = score
            d.results.append(hyp)

            label, id_conf = self._classify_crop(frame, (x1, y1, x2, y2))
            if label is not None:
                id_hyp = ObjectHypothesisWithPose()
                id_hyp.hypothesis.class_id = label
                id_hyp.hypothesis.score = id_conf
                d.results.append(id_hyp)
            identities.append((label, id_conf))

            out.detections.append(d)
        self.det_pub.publish(out)

        if out.detections:
            self._handle_sighting(frame, identities)

        if self.ann_pub is not None:
            annotated = r.plot()  # BGR np.ndarray with boxes drawn
            for (label, id_conf), b in zip(identities, r.boxes):
                if label is None:
                    continue
                x1, y1 = b.xyxy[0][:2].tolist()
                text = f"{label} {id_conf:.2f}"
                cv2.putText(annotated, text, (int(x1), max(0, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            ann_msg.header = msg.header
            self.ann_pub.publish(ann_msg)

        self._frames += 1
        now = time.monotonic()
        if now - self._t0 >= self.log_fps_every:
            fps = self._frames / (now - self._t0)
            self.get_logger().info(f"inference {fps:.1f} FPS, {len(out.detections)} det")
            self._frames = 0
            self._t0 = now

    def _handle_sighting(self, frame, identities=None):
        """Called once per frame that contains >=1 cat detection.

        Beeps once per NEW sighting (a gap of sighting_gap_sec with no cat
        counts as "new"), and saves a timestamped frame for the Phase 5b
        dataset -- immediately on a new sighting, then throttled to at most
        one more every detection_save_interval_sec while it continues, so a
        cat napping in frame for ten minutes doesn't fill the disk with
        near-identical images.
        """
        now = time.monotonic()
        is_new_sighting = (
            self._last_seen_time is None
            or (now - self._last_seen_time) > self.sighting_gap_sec
        )
        self._last_seen_time = now

        label = None
        if identities:
            named = [(l, c) for l, c in identities if l is not None]
            if named:
                label = max(named, key=lambda lc: lc[1])[0]

        if is_new_sighting:
            self.get_logger().info(
                f"Cat sighting started" + (f" ({label})" if label else ""))
            if self.beep_on_detection:
                self._beep()
            if self.save_on_detection:
                self._save_frame(frame, label)
                self._last_saved_time = now
        elif self.save_on_detection and (
            self._last_saved_time is None
            or (now - self._last_saved_time) >= self.detection_save_interval_sec
        ):
            self._save_frame(frame, label)
            self._last_saved_time = now

    def _beep(self):
        on = Bool()
        on.data = True
        self.beep_pub.publish(on)
        # Buzzer driver just toggles a GPIO on/off (Mcnamu_driver_X3.py); turn
        # it back off after one short pulse via a plain Python timer so a
        # sighting doesn't leave it latched on. threading.Timer fires off the
        # rclpy executor thread, but a simple publish() call from it is fine.
        threading.Timer(self.beep_duration_sec, self._beep_off).start()

    def _beep_off(self):
        off = Bool()
        off.data = False
        self.beep_pub.publish(off)

    def _save_frame(self, frame, label=None):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # Label suffix comes after the timestamp so cat_*.jpg sorting/eviction
        # in _enforce_retention (chronological via plain name sort) still works.
        name = f"cat_{ts}_{label}.jpg" if label else f"cat_{ts}.jpg"
        fp = os.path.join(self.detection_image_dir, name)
        if cv2.imwrite(fp, frame):
            self.get_logger().info(f"Saved detection frame: {fp}")
            self._enforce_retention()

    def _enforce_retention(self):
        # Filenames are cat_<timestamp>.jpg, so a plain name sort is also a
        # chronological sort -- oldest files come first, no stat() calls needed.
        files = sorted(glob.glob(os.path.join(self.detection_image_dir, "cat_*.jpg")))
        excess = len(files) - self.max_detection_images
        for fp in files[:max(excess, 0)]:
            try:
                os.remove(fp)
            except OSError as e:
                self.get_logger().warning(f"Failed to evict {fp}: {e}")


def main():
    rclpy.init()
    node = CatDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
