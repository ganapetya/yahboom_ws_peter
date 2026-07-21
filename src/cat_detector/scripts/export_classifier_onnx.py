#!/usr/bin/env python3
"""Export the trained cat_id_classifier.pt (MobileNetV2 head, see
train_classifier.py) to ONNX so cat_detector_cpp can load it via OpenCV's
cv::dnn (same route already used for yolov8n.onnx, §14b of
phase5-status.md -- cat_detector_cpp deliberately has no libtorch
dependency, only cv::dnn, so a PyTorch .pt checkpoint can't be loaded
directly in C++).

Usage:
    python3 export_classifier_onnx.py
    python3 export_classifier_onnx.py --checkpoint /path/to/cat_id_classifier.pt \\
        --output /path/to/cat_id_classifier.onnx
"""
import argparse

import torch
import torch.nn as nn
from torchvision import models


def build_model(classes):
    net = models.mobilenet_v2(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, len(classes))
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint",
        default="/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/cat_detector/models/cat_id_classifier.pt")
    ap.add_argument(
        "--output",
        default="/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/cat_detector_cpp/models/cat_id_classifier.onnx")
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    print(f"Loaded checkpoint, classes={classes}, test_acc={checkpoint.get('test_acc')}")

    net = build_model(classes)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    # Fixed batch=1, 224x224x3 input -- matches the eval-time preprocessing in
    # train_classifier.py (Resize(256) + CenterCrop(224)) and what
    # detector_node.cpp will replicate manually before feeding cv::dnn.
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        net, dummy, args.output,
        input_names=["input"], output_names=["output"],
        opset_version=12,
    )
    print(f"Exported to {args.output}")
    print(f"Class order (index -> label): {list(enumerate(classes))}")
    print("IMPORTANT: cat_detector_cpp_params.yaml's classifier_classes must "
          "list these in this exact order.")


if __name__ == "__main__":
    main()
