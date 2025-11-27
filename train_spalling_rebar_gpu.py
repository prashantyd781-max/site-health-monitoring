"""
YOLOv8 Segmentation Training - Spalling & Exposed Rebar Dataset
GPU-Accelerated training using Apple M4 Metal Performance Shaders (MPS)
"""

from ultralytics import YOLO
import torch

def train_spalling_rebar_gpu():
    """Train YOLOv8 SEGMENTATION model on Spalling & Exposed Rebar dataset with GPU"""
    
    print("\n" + "🚀" * 40)
    print("STARTING YOLOV8 SEGMENTATION TRAINING - SPALLING & REBAR")
    print("🚀" * 40 + "\n")
    
    # Check GPU availability
    print("🔍 Checking GPU availability...")
    if torch.backends.mps.is_available():
        print("✅ Apple M4 GPU (MPS) is available and will be used!")
        device = 'mps'
    else:
        print("⚠️  GPU not available, falling back to CPU")
        device = 'cpu'
    
    print(f"\n📊 Dataset Information:")
    print("   Name: Spalling and Exposed Rebar")
    print("   Training Images: 2,997")
    print("   Validation Images: 120")
    print("   Test Images: 108")
    print("   Classes: 2 (exposed_rebar, spalling)")
    print("   Task: Instance Segmentation (Polygon Masks)")
    print(f"   Device: {device.upper()}")
    print("\n" + "=" * 80 + "\n")
    
    # Initialize YOLOv8 SEGMENTATION model
    print("🔧 Loading YOLOv8n-seg model...")
    model = YOLO('yolov8n-seg.pt')
    
    # Dataset path
    data_yaml = 'segmentation_model/Spalling and exposed rebar.v1i.yolov8/data.yaml'
    
    print("🏋️  Starting training...\n")
    
    # Training parameters optimized for GPU - batch=4 to prevent thermal throttling
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=4,  # Reduced to 4 to prevent GPU overheating/thermal throttling
        name='yolov8_spalling_rebar_final',
        patience=20,
        save=True,
        device=device,
        project='runs/segment',
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        plots=True,
        verbose=True,
        rect=False,  # Disable rectangular training to avoid shape mismatches
        close_mosaic=10  # Close mosaic augmentation after 10 epochs for stability
    )
    
    print("\n" + "=" * 80)
    print("🎉 SPALLING & REBAR SEGMENTATION TRAINING COMPLETED! 🎉")
    print("=" * 80)
    print("\n📂 Trained Model Location:")
    print(f"   runs/segment/yolov8_spalling_rebar_final/weights/best.pt")
    print("\n📊 Training Results:")
    print(f"   Results CSV: runs/segment/yolov8_spalling_rebar_final/results.csv")
    print(f"   Training Curves: runs/segment/yolov8_spalling_rebar_final/results.png")
    print("\n💾 Model Details:")
    print("   Dataset: Spalling and Exposed Rebar")
    print("   Classes: exposed_rebar, spalling")
    print("   Type: Instance Segmentation")
    print(f"   Device: {device.upper()}")
    print("=" * 80 + "\n")
    
    return results

if __name__ == "__main__":
    # Train the segmentation model
    train_spalling_rebar_gpu()

