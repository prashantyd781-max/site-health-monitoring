import os
from ultralytics import YOLO

# 1. Load your trained segmentation model (YOLOv11 compatible)
model = YOLO("segmentation_model/weights/best.pt")  # Updated path to correct segmentation model

# 2. Path to your new dataset (all unlabeled images)
source_folder = "new_crack_images"   # put your 30k images here

# 3. Output folders (YOLO format)
output_images = "autolabel_dataset/train/images"
output_labels = "autolabel_dataset/train/labels"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# 4. Run prediction on all images
results = model.predict(
    source=source_folder,
    save=False,              # we don't need images with boxes, just labels
    save_txt=True,           # save YOLO format labels
    save_conf=False,         # don't save confidences
    project="autolabel_dataset",
    name="train",            # keeps labels inside /train/labels
    exist_ok=True
)

print("✅ Auto-labeling complete!")
print(f"Check your labels inside: {output_labels}")
