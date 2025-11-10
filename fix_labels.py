import os
import argparse

def fix_labels_in_folder(folder):
    count = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith('.txt'):
                path = os.path.join(root, f)
                with open(path, 'r') as file:
                    lines = file.readlines()

                new_lines = []
                changed = False
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                        if 1 <= cls_id <= 9:
                            cls_id -= 1  # shift to 0-based
                            parts[0] = str(cls_id)
                            changed = True
                        new_lines.append(' '.join(parts) + '\n')
                    except ValueError:
                        # skip malformed lines
                        continue

                if changed:
                    with open(path, 'w') as file:
                        file.writelines(new_lines)
                        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix YOLO label IDs to zero-based.")
    parser.add_argument("--root", required=True, help="Path to dataset root containing train/val/test folders.")
    args = parser.parse_args()

    total_fixed = 0
    for sub in ['train', 'val', 'test']:
        label_dir = os.path.join(args.root, sub, 'labels')
        if os.path.exists(label_dir):
            fixed = fix_labels_in_folder(label_dir)
            print(f"✅ Fixed {fixed} files in {label_dir}")
            total_fixed += fixed
        else:
            print(f"⚠️ No labels folder found in {label_dir}")

    print(f"\n🎯 Done! Total label files updated: {total_fixed}")
