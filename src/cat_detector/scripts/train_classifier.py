#!/usr/bin/env python3
"""Phase 5b: fine-tune a lightweight white-cat/brown-cat classifier.

Input: ~/cat_patrol_data/dataset/{white,brown}/*.jpg -- YOLO cat crops
produced by collect_dataset.py.

Model: MobileNetV2 (ImageNet-pretrained), backbone frozen, only the final
classifier head fine-tuned -- appropriate for a dataset this small (dozens,
not thousands, of images per class) where fine-tuning the whole network
would just overfit.

Augmentation is deliberately mild and, crucially, has NO hue/saturation
jitter: color is the actual signal distinguishing the two cats, so
randomizing it would train the model to ignore the one feature that matters.

Split: group-aware so near-duplicate frames from the same sighting (the
robot saves one frame every few seconds while a cat lingers, per
phase5-status.md §2/§6c) never land in both train and test -- that would
silently inflate the reported test accuracy. Frames whose filename carries
the robot's own timestamp format are grouped into 60s sighting buckets;
other photos (phone photos) are each their own group.

Usage:
    python3 train_classifier.py
    python3 train_classifier.py --epochs 30 --data-dir ~/cat_patrol_data/dataset
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

CLASSES = ["brown", "white"]  # alphabetical -> stable index mapping
TS_RE = re.compile(r"cat_(\d{8})_(\d{6})_(\d{3})")
SIGHTING_BUCKET_SEC = 60


def group_key(filename):
    m = TS_RE.search(filename)
    if not m:
        return filename  # phone photo: its own group, nothing to dedup against
    date_s, time_s, _ = m.groups()
    dt = datetime.strptime(date_s + time_s, "%Y%m%d%H%M%S")
    bucket = int(dt.timestamp()) // SIGHTING_BUCKET_SEC
    return f"session_{date_s}_{bucket}"


def list_class_files(data_dir, cls):
    d = os.path.join(data_dir, cls)
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def group_split(files, seed, train_frac=0.70, val_frac=0.15):
    groups = defaultdict(list)
    for f in files:
        groups[group_key(f)].append(f)
    group_ids = list(groups.keys())
    random.Random(seed).shuffle(group_ids)

    n = len(group_ids)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac)) if n > 2 else 0
    train_g = group_ids[:n_train]
    val_g = group_ids[n_train:n_train + n_val]
    test_g = group_ids[n_train + n_val:]
    if not test_g:  # guarantee a non-empty test split even for a tiny class
        test_g = [val_g.pop()] if val_g else [train_g.pop()]

    def flatten(gs):
        out = []
        for g in gs:
            out.extend(groups[g])
        return out

    return flatten(train_g), flatten(val_g), flatten(test_g)


class CropDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_model(device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return model.to(device)


def evaluate(model, loader, device):
    model.eval()
    n_correct, n_total = 0, 0
    confusion = [[0, 0], [0, 0]]  # [true][pred]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            for t, p in zip(y.tolist(), pred.tolist()):
                confusion[t][p] += 1
            n_correct += (pred == y).sum().item()
            n_total += y.size(0)
    acc = n_correct / n_total if n_total else 0.0
    return acc, confusion


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="~/cat_patrol_data/dataset")
    ap.add_argument("--output", default="~/yahboomcar_ros2_ws/yahboomcar_ws/src/cat_detector/models/cat_id_classifier.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    out_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_samples, val_samples, test_samples = [], [], []
    class_counts = {}
    for cls in CLASSES:
        files = list_class_files(data_dir, cls)
        class_counts[cls] = len(files)
        tr, va, te = group_split(files, seed=args.seed)
        label = CLASSES.index(cls)
        train_samples += [(os.path.join(data_dir, cls, f), label) for f in tr]
        val_samples += [(os.path.join(data_dir, cls, f), label) for f in va]
        test_samples += [(os.path.join(data_dir, cls, f), label) for f in te]

    print(f"Classes: {CLASSES}, counts: {class_counts}")
    print(f"Split -> train {len(train_samples)}, val {len(val_samples)}, test {len(test_samples)}")

    # Class weights to counter the white/brown count imbalance.
    counts_by_idx = [class_counts[c] for c in CLASSES]
    weights = torch.tensor([1.0 / c for c in counts_by_idx], dtype=torch.float32)
    weights = weights / weights.sum() * len(CLASSES)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # no hue/saturation on purpose
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(CropDataset(train_samples, train_tf), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CropDataset(val_samples, eval_tf), batch_size=args.batch_size)
    test_loader = DataLoader(CropDataset(test_samples, eval_tf), batch_size=args.batch_size)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_acc, _ = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d}  train_loss {total_loss / len(train_samples):.4f}  val_acc {val_acc:.3f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, confusion = evaluate(model, test_loader, device)

    print(f"\nBest val_acc: {best_val_acc:.3f}")
    print(f"Test acc: {test_acc:.3f}")
    print("Confusion matrix (rows=true, cols=pred), classes =", CLASSES)
    for i, row in enumerate(confusion):
        print(f"  {CLASSES[i]:>6}: {row}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "arch": "mobilenet_v2",
        "test_acc": test_acc,
        "confusion": confusion,
    }, out_path)
    print(f"\nSaved classifier to {out_path}")

    meta_path = out_path.replace(".pt", ".json")
    with open(meta_path, "w") as f:
        json.dump({"classes": CLASSES, "test_acc": test_acc, "confusion": confusion,
                    "class_counts": class_counts}, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
