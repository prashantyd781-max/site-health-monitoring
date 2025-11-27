import os
import shutil
import random
from pathlib import Path
import yaml

# Paths
raw_dataset = "DATA_Maguire_20180517_ALL"
output_dir = "dataset"

# Classes
classes = ["CD", "UD"]

# Train/val split ratio
train_split = 0.8

# Create output directories
for split in ["train", "val"]:
    for sub in ["images", "labels"]:
        os.makedirs(f"{output_dir}/{sub}/{split}", exist_ok=True)

# Process each class
for class_id, class_name in enumerate(classes):
    img_dir = Path(raw_dataset) / class_name
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    random.shuffle(images)
    split_idx = int(len(images) * train_split)
    train_images, val_images = images[:split_idx], images[split_idx:]

    for split, split_images in zip(["train", "val"], [train_images, val_images]):
        for img_path in split_images:
            # Copy image
            dst_img = Path(output_dir) / "images" / split / img_path.name
            shutil.copy(img_path, dst_img)

            # Create YOLO label (full image assigned to class)
            dst_lbl = Path(output_dir) / "labels" / split / (img_path.stem + ".txt")
            with open(dst_lbl, "w") as f:
                f.write(f"{class_id} 0 0 1 1\n")  # full image as bounding box

# Write data.yaml
data_yaml = {
    "train": str(Path(output_dir) / "images/train"),
    "val": str(Path(output_dir) / "images/val"),
    "nc": len(classes),
    "names": classes
}

with open(f"{output_dir}/data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("✅ Dataset prepared at:", output_dir)
