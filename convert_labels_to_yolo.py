#!/usr/bin/env python3
"""
convert_labels_to_yolo.py

Usage examples:
    python3 convert_labels_to_yolo.py \
        --data-root /path/to/dataset \
        --out-root /path/to/dataset_converted \
        --train-dir train \
        --val-dir validation \
        --test-dir test

What it does:
 - Reads original label .txt files in train/labels and validation/labels.
 - For each annotation-line: extracts tokens, maps class name -> encoded id,
   takes tokens 5,6,7,8 as x1,y1,x2,y2 (1-based counting from your example),
   converts to YOLO (x_center,y_center,w,h) normalized by image size.
 - Writes new .txt files into out_root/{train,val,test}/labels
 - For test images (no original labels) it creates empty .txt files so YOLO tools
   that expect a label file per image will find one.
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm

CLASS_MAP = {
    "Car": 1,
    "Van": 2,
    "Truck": 3,
    "Pedestrian": 4,
    "Person_sitting": 5,
    "Cyclist": 6,
    "Tram": 7,
    "Misc": 8,
    "DontCare": 9,
}

def convert_line_to_yolo(tokens, img_w, img_h, class_map, zero_based=False):
    # tokens: [classname, t1, t2, t3, t4, t5, t6, t7, ...]
    # from user's sample the 5th,6th,7th,8th elements (1-indexed):
    # tokens[4], tokens[5], tokens[6], tokens[7] are x1,y1,x2,y2
    if len(tokens) < 8:
        return None  # malformed
    cls_name = tokens[0]
    if cls_name not in class_map:
        return None
    try:
        x1 = float(tokens[4])
        y1 = float(tokens[5])
        x2 = float(tokens[6])
        y2 = float(tokens[7])
    except Exception:
        return None

    # discard boxes that are degenerate or clearly invalid
    if x2 <= x1 or y2 <= y1:
        return None
    # Optionally clip to image bounds
    x1 = max(0.0, min(x1, img_w))
    x2 = max(0.0, min(x2, img_w))
    y1 = max(0.0, min(y1, img_h))
    y2 = max(0.0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        return None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    # normalize
    cx_n = cx / img_w
    cy_n = cy / img_h
    w_n = w / img_w
    h_n = h / img_h

    class_id = class_map[cls_name]
    if zero_based:
        class_id = class_id - 1

    # ensure values in [0,1]
    cx_n = min(max(cx_n, 0.0), 1.0)
    cy_n = min(max(cy_n, 0.0), 1.0)
    w_n = min(max(w_n, 0.0), 1.0)
    h_n = min(max(h_n, 0.0), 1.0)

    return f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"

def process_split(images_dir, labels_dir, out_labels_dir, class_map, zero_based=False):
    os.makedirs(out_labels_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for img_name in tqdm(image_files, desc=f"Processing {os.path.basename(images_dir)}"):
        img_path = os.path.join(images_dir, img_name)
        base, _ = os.path.splitext(img_name)
        src_label_path = os.path.join(labels_dir, base + ".txt") if labels_dir is not None else None
        out_label_path = os.path.join(out_labels_dir, base + ".txt")

        # if no labels exist for image (like test), create empty output label
        if src_label_path is None or not os.path.exists(src_label_path):
            # create empty .txt so YOLO tooling sees a label file per image
            open(out_label_path, "w").close()
            continue

        # open image to get size
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            print(f"Warning: could not open image {img_path}: {e}. Skipping.")
            continue

        converted_lines = []
        with open(src_label_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                yolo_line = convert_line_to_yolo(tokens, img_w, img_h, class_map, zero_based=zero_based)
                if yolo_line:
                    converted_lines.append(yolo_line)

        # write converted lines (if none, write empty file)
        with open(out_label_path, "w") as of:
            for ln in converted_lines:
                of.write(ln + "\n")

def main():
    parser = argparse.ArgumentParser(description="Convert custom .txt labels to YOLO format using tokens 5-8 as x1,y1,x2,y2")
    parser.add_argument("--data-root", required=True, help="Root folder containing train/validation/test directories")
    parser.add_argument("--out-root", required=True, help="Where to write converted dataset (keeps original intact)")
    parser.add_argument("--train-dir", default="train", help="Name of train folder under data-root")
    parser.add_argument("--val-dir", default="validation", help="Name of validation folder under data-root")
    parser.add_argument("--test-dir", default="test", help="Name of test folder under data-root")
    parser.add_argument("--zero-based", action="store_true", help="Use zero-based class ids (0..8). By default uses mapping 1..9")
    parser.add_argument("--include-dontcare", action="store_true",
                        help="Include DontCare lines (class id 9). Non-included by default? NOTE: default is to include.")
    args = parser.parse_args()

    # NOTE: by default we include DontCare because your mapping included it.
    # If you want to ignore DontCare boxes, use filtering logic here.
    class_map = CLASS_MAP.copy()

    data_root = args.data_root
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    # TRAIN
    train_images_dir = os.path.join(data_root, args.train_dir, "images")
    train_labels_dir = os.path.join(data_root, args.train_dir, "labels")
    out_train_labels = os.path.join(out_root, args.train_dir, "labels")
    if os.path.isdir(train_images_dir):
        process_split(train_images_dir, train_labels_dir, out_train_labels, class_map, zero_based=args.zero_based)
    else:
        print(f"Train images folder not found: {train_images_dir}")

    # VALIDATION
    val_images_dir = os.path.join(data_root, args.val_dir, "images")
    val_labels_dir = os.path.join(data_root, args.val_dir, "labels")
    out_val_labels = os.path.join(out_root, args.val_dir, "labels")
    if os.path.isdir(val_images_dir):
        process_split(val_images_dir, val_labels_dir, out_val_labels, class_map, zero_based=args.zero_based)
    else:
        print(f"Validation images folder not found: {val_images_dir}")

    # TEST (create empty label files if no labels exist)
    test_images_dir = os.path.join(data_root, args.test_dir, "images")
    # test original labels dir (if any)
    test_labels_dir = os.path.join(data_root, args.test_dir, "labels")
    out_test_labels = os.path.join(out_root, args.test_dir, "labels")
    if os.path.isdir(test_images_dir):
        # if test has labels, pass the labels dir; otherwise pass None so empty files are created
        src_labels_dir = test_labels_dir if os.path.isdir(test_labels_dir) else None
        process_split(test_images_dir, src_labels_dir, out_test_labels, class_map, zero_based=args.zero_based)
    else:
        print(f"Test images folder not found: {test_images_dir}")

    print("Conversion finished. Converted labels are in:", out_root)

if __name__ == "__main__":
    main()
