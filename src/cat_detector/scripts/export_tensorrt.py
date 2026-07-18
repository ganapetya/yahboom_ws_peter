#!/usr/bin/env python3
"""Export yolov8n.pt -> TensorRT .engine and benchmark FPS before/after.

Run manually on the Jetson after the .pt model works live via detector_node:
    python3 src/cat_detector/scripts/export_tensorrt.py
"""
import time

import numpy as np
from ultralytics import YOLO

PT = "/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/cat_detector/models/yolov8n.pt"
IMGSZ = 640
DEVICE = "cuda:0"


def bench(model, n=100):
    dummy = (np.random.rand(IMGSZ, IMGSZ, 3) * 255).astype("uint8")
    for _ in range(10):
        model.predict(dummy, imgsz=IMGSZ, device=DEVICE, verbose=False)
    t0 = time.monotonic()
    for _ in range(n):
        model.predict(dummy, imgsz=IMGSZ, device=DEVICE, verbose=False)
    return n / (time.monotonic() - t0)


if __name__ == "__main__":
    pt = YOLO(PT)
    print(f"PyTorch .pt : {bench(pt):.1f} FPS")

    # half=True (FP16) is the usual Orin win; int8 needs a calibration set.
    pt.export(format="engine", imgsz=IMGSZ, half=True, device=0)
    engine = YOLO(PT.replace(".pt", ".engine"))
    print(f"TensorRT .engine (fp16): {bench(engine):.1f} FPS")
