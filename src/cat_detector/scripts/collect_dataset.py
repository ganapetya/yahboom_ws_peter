#!/usr/bin/env python3
"""Phase 5b flywheel: crop cats out of archived frames into dataset/unsorted/.

Run periodically (by hand) against the persistent capture directories.
Source frames are never modified or deleted -- this only reads them and
writes crops elsewhere, so it's safe to re-run repeatedly (already-cropped
source files are skipped).

Usage:
    python3 collect_dataset.py \\
        --source ~/cat_patrol_data/detections ~/cat_patrol_data/captures \\
        --dataset-dir ~/cat_patrol_data/dataset \\
        --model /path/to/yolov8n.pt

After running, manually sort dataset/unsorted/*.jpg into
dataset/cat_A/, dataset/cat_B/, dataset/none/ before fine-tuning (§9 Step C
in phase5-status.md).
"""
import argparse
import glob
import os

import cv2
from ultralytics import YOLO

CAT_CLASS_ID = 15
PAD_FRAC = 0.15  # pad the crop by this fraction of box size on each side


def crop_with_padding(frame, x1, y1, x2, y2, pad_frac):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px))
    y2 = min(h, int(y2 + py))
    return frame[y1:y2, x1:x2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", nargs="+", required=True,
                     help="One or more directories of source JPEGs to scan")
    ap.add_argument("--dataset-dir", required=True,
                     help="Output dataset root; crops land in <dataset-dir>/unsorted/")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights to crop with")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.45)
    args = ap.parse_args()

    unsorted_dir = os.path.join(os.path.expanduser(args.dataset_dir), "unsorted")
    os.makedirs(unsorted_dir, exist_ok=True)

    model = YOLO(args.model)

    already_done = {
        fn.split("__crop")[0]
        for fn in os.listdir(unsorted_dir)
        if "__crop" in fn
    }

    n_scanned = 0
    n_crops = 0
    for source_dir in args.source:
        source_dir = os.path.expanduser(source_dir)
        for fp in sorted(glob.glob(os.path.join(source_dir, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(fp))[0]
            if stem in already_done:
                continue
            n_scanned += 1

            frame = cv2.imread(fp)
            if frame is None:
                continue
            results = model.predict(
                frame, conf=args.conf, classes=[CAT_CLASS_ID],
                device=args.device, verbose=False)
            boxes = results[0].boxes
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                crop = crop_with_padding(frame, x1, y1, x2, y2, PAD_FRAC)
                if crop.size == 0:
                    continue
                out_fp = os.path.join(unsorted_dir, f"{stem}__crop{i}.jpg")
                cv2.imwrite(out_fp, crop)
                n_crops += 1

    print(f"Scanned {n_scanned} new source frames, wrote {n_crops} crops to {unsorted_dir}")
    print("Next: sort unsorted/*.jpg into cat_A/, cat_B/, none/ before fine-tuning.")


if __name__ == "__main__":
    main()
