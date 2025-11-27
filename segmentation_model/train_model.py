from ultralytics import YOLO
import os

# ✅ Path to pretrained weights
weights_path = "segmentation_model/weights/best.pt"
dataset_yaml = "dataset/data.yaml"

# Safety checks
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"❌ Could not find weights at {weights_path}")
if not os.path.exists(dataset_yaml):
    raise FileNotFoundError(f"❌ Could not find dataset config at {dataset_yaml}")

# Load model
print(f"Loading model from: {weights_path}")
model = YOLO(weights_path)

# 🚨 Force a fresh run (disable resume, set custom run name)
results = model.train(
    data=dataset_yaml,
    epochs=100,
    imgsz=256,
    batch=16,
    resume=False,
    name="custom_train"   # forces new folder: runs/detect/custom_train
)

print("✅ Training complete! Check 'runs/detect/custom_train' for results.")
