#!/usr/bin/env python3
"""Phase 5b flywheel: crop cats out of raw photos into a labeled dataset.

Two kinds of sources:
  - LABELED dirs (~/cats/white, ~/cats/brown): every photo is known to be
    that one cat, so crops are written straight into dataset/<label>/.
  - UNLABELED flywheel dirs (patrol waypoint captures + detector-triggered
    saves): every frame *might* have a cat but we don't know which one, so
    crops land in dataset/unsorted/ for a manual sort pass into white/ or
    brown/ (per phase5-status.md §9 Step C).

Source frames are never modified or deleted -- this only reads them and
writes crops elsewhere, so it's safe to re-run repeatedly (already-cropped
source files are skipped via the __crop<n> suffix marker).

If YOLO finds no "cat" box in a LABELED photo (common on tight phone
close-ups that fill the whole frame), the original image is copied through
uncropped rather than silently dropped -- every labeled photo is known to
contain that cat, so we never want to lose one. Unlabeled flywheel photos
with no detection are just skipped (most waypoint captures have no cat at
all -- that's expected, see phase5-status.md §2).

Usage (defaults match the Phase 5b directory layout in phase5-status.md):
    python3 collect_dataset.py
    python3 collect_dataset.py --labeled white=~/cats/white brown=~/cats/brown \\
        --unlabeled ~/cat_patrol_data/captures ~/cat_patrol_data/detections \\
        --dataset-dir ~/cat_patrol_data/dataset --model /path/to/yolov8n.pt
"""
import argparse
import glob
import os

import cv2
from ultralytics import YOLO

CAT_CLASS_ID = 15
PAD_FRAC = 0.15  # pad the crop by this fraction of box size on each side
IMG_EXTS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")

DEFAULT_LABELED = {
    "white": "~/cats/white",
    "brown": "~/cats/brown",
}
DEFAULT_UNLABELED = [
    "~/cat_patrol_data/captures",
    "~/cat_patrol_data/detections",
    "~/cat_patrol_data/detections_cpp",
]


def list_images(d):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(d, ext)))
    return sorted(set(files))


def crop_with_padding(frame, x1, y1, x2, y2, pad_frac):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px))
    y2 = min(h, int(y2 + py))
    return frame[y1:y2, x1:x2]


def already_done_stems(out_dir):
    return {
        fn.split("__crop")[0]
        for fn in os.listdir(out_dir)
        if "__crop" in fn or fn.startswith("passthrough__")
    }


def process_dir(model, source_dir, out_dir, conf, device, passthrough_if_empty, counters):
    source_dir = os.path.expanduser(source_dir)
    if not os.path.isdir(source_dir):
        print(f"  (skip, not a directory: {source_dir})")
        return

    done = already_done_stems(out_dir)
    images = list_images(source_dir)
    print(f"  {len(images)} photos in {source_dir}")

    for fp in images:
        stem = os.path.splitext(os.path.basename(fp))[0]
        if stem in done:
            continue

        frame = cv2.imread(fp)
        if frame is None:
            counters["unreadable"] += 1
            print(f"    [unreadable] {fp}")
            continue

        results = model.predict(
            frame, conf=conf, classes=[CAT_CLASS_ID], device=device, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            if passthrough_if_empty:
                out_fp = os.path.join(out_dir, f"passthrough__{stem}.jpg")
                cv2.imwrite(out_fp, frame)
                counters["no_detection_passthrough"] += 1
            else:
                counters["no_detection_skipped"] += 1
            continue

        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            crop = crop_with_padding(frame, x1, y1, x2, y2, PAD_FRAC)
            if crop.size == 0:
                continue
            out_fp = os.path.join(out_dir, f"{stem}__crop{i}.jpg")
            cv2.imwrite(out_fp, crop)
            counters["cropped"] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled", nargs="*", default=[f"{k}={v}" for k, v in DEFAULT_LABELED.items()],
                     help="label=dir pairs, e.g. white=~/cats/white brown=~/cats/brown")
    ap.add_argument("--unlabeled", nargs="*", default=DEFAULT_UNLABELED,
                     help="Flywheel directories with unknown-cat photos -> dataset/unsorted/")
    ap.add_argument("--dataset-dir", default="~/cat_patrol_data/dataset")
    ap.add_argument("--model", default="/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/cat_detector/models/yolov8n.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.25,
                     help="Lower than the live detector's 0.45 -- still photos are generally "
                          "sharper/less noisy than a live camera feed")
    args = ap.parse_args()

    dataset_dir = os.path.expanduser(args.dataset_dir)
    print(f"Loading {args.model} ...")
    model = YOLO(args.model)

    counters = {
        "cropped": 0, "no_detection_passthrough": 0,
        "no_detection_skipped": 0, "unreadable": 0,
    }

    for pair in args.labeled:
        label, src_dir = pair.split("=", 1)
        out_dir = os.path.join(dataset_dir, label)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n== labeled '{label}' ==")
        process_dir(model, src_dir, out_dir, args.conf, args.device,
                     passthrough_if_empty=True, counters=counters)

    unsorted_dir = os.path.join(dataset_dir, "unsorted")
    os.makedirs(unsorted_dir, exist_ok=True)
    for src_dir in args.unlabeled:
        print(f"\n== unlabeled flywheel: {src_dir} ==")
        process_dir(model, src_dir, unsorted_dir, args.conf, args.device,
                     passthrough_if_empty=False, counters=counters)

    print("\n=== Summary ===")
    for k, v in counters.items():
        print(f"  {k}: {v}")
    for label in list(dict(p.split("=", 1) for p in args.labeled).keys()) + ["unsorted"]:
        d = os.path.join(dataset_dir, label)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"dataset/{label}: {n} files")
    print("\nNext: manually sort dataset/unsorted/*.jpg into dataset/white/ or "
          "dataset/brown/ before fine-tuning (phase5-status.md §9 Step C).")


if __name__ == "__main__":
    main()
