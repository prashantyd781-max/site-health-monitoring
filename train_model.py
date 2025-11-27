from ultralytics import YOLO
import os

# Path to your professor's pretrained model
weights_path = "segmentation_model/weights/best.pt"

# Check if the file exists
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"❌ Could not find weights at {weights_path}")

# Load model
model = YOLO(weights_path)

# Train on your prepared dataset
results = model.train(
    data="dataset/data.yaml",  # dataset config file
    epochs=100,                # number of epochs
    imgsz=256,                 # image size
    batch=16                   # batch size (reduce if memory issues)
)

print("✅ Training complete! Check 'runs/detect/train' for results.")
