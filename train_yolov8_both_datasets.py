"""
YOLOv8 Training Script for Multiple Datasets
Trains YOLOv8 models on:
1. Crack Augmented Dataset (1 class: Crack) - DETECTION
2. Spalling and Exposed Rebar Dataset (2 classes: exposed_rebar, spalling) - SEGMENTATION
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_crack_augmented():
    """Train YOLOv8 DETECTION model on the crack_augmented dataset"""
    print("=" * 80)
    print("TRAINING YOLOV8 DETECTION MODEL ON CRACK AUGMENTED DATASET")
    print("=" * 80)
    
    # Initialize YOLOv8 DETECTION model (not segmentation)
    model = YOLO('yolov8n.pt')
    
    # Dataset path
    data_yaml = 'segmentation_model/crack_augmented/data.yaml'
    
    # Training parameters
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8_crack_detection_gpu',
        patience=20,
        save=True,
        device='mps',  # Using Apple M4 GPU
        project='runs/detect',
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
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("CRACK AUGMENTED DETECTION TRAINING COMPLETED!")
    print(f"Best model saved at: runs/detect/yolov8_crack_detection_gpu/weights/best.pt")
    print("=" * 80 + "\n")
    
    return results

def train_spalling_rebar():
    """Train YOLOv8 SEGMENTATION model on the Spalling and Exposed Rebar dataset"""
    print("=" * 80)
    print("TRAINING YOLOV8 SEGMENTATION MODEL ON SPALLING AND EXPOSED REBAR DATASET")
    print("=" * 80)
    
    # Initialize YOLOv8 SEGMENTATION model
    model = YOLO('yolov8n-seg.pt')
    
    # Dataset path
    data_yaml = 'segmentation_model/Spalling and exposed rebar.v1i.yolov8/data.yaml'
    
    # Training parameters
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8_spalling_rebar_segmentation_gpu',
        patience=20,
        save=True,
        device='mps',  # Using Apple M4 GPU
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
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("SPALLING AND EXPOSED REBAR SEGMENTATION TRAINING COMPLETED!")
    print(f"Best model saved at: runs/segment/yolov8_spalling_rebar_segmentation_gpu/weights/best.pt")
    print("=" * 80 + "\n")
    
    return results

def train_combined():
    """Train on both datasets sequentially with appropriate models"""
    print("\n" + "🚀" * 40)
    print("STARTING YOLOV8 TRAINING ON BOTH DATASETS")
    print("🚀" * 40 + "\n")
    
    print("📋 Dataset Information:")
    print("   1. Crack Augmented: 1620 train, 100 valid images")
    print("      Format: Detection (bounding boxes)")
    print("      Model: YOLOv8n Detection")
    print()
    print("   2. Spalling & Rebar: 2997 train, 120 valid images")
    print("      Format: Segmentation (polygon masks)")
    print("      Model: YOLOv8n Segmentation")
    print("\n" + "=" * 80 + "\n")
    
    # Check if YOLO model weights exist
    if not os.path.exists('yolov8n.pt'):
        print("Downloading YOLOv8n detection pretrained weights...")
        model = YOLO('yolov8n.pt')  # This will download automatically
    
    if not os.path.exists('yolov8n-seg.pt'):
        print("Downloading YOLOv8n-seg segmentation pretrained weights...")
        model = YOLO('yolov8n-seg.pt')  # This will download automatically
    
    # Train on crack augmented dataset (DETECTION)
    print("\n[1/2] Training Detection Model on Crack Augmented Dataset...")
    crack_results = train_crack_augmented()
    
    # Train on spalling and rebar dataset (SEGMENTATION)
    print("\n[2/2] Training Segmentation Model on Spalling and Exposed Rebar Dataset...")
    spalling_results = train_spalling_rebar()
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 ALL TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 80)
    print("\nTrained Models:")
    print("1. Crack Detection Model:")
    print("   📂 Location: runs/detect/yolov8_crack_detection_gpu/weights/best.pt")
    print("   📊 Dataset: 1620 train, 100 valid, 100 test images")
    print("   🏷️  Classes: Crack")
    print("   🔧 Type: Object Detection (Bounding Boxes)")
    print("   ⚡ Device: Apple M4 GPU (MPS)")
    print()
    print("2. Spalling & Rebar Segmentation Model:")
    print("   📂 Location: runs/segment/yolov8_spalling_rebar_segmentation_gpu/weights/best.pt")
    print("   📊 Dataset: 2997 train, 120 valid, 108 test images")
    print("   🏷️  Classes: exposed_rebar, spalling")
    print("   🔧 Type: Instance Segmentation (Polygon Masks)")
    print("   ⚡ Device: Apple M4 GPU (MPS)")
    print("=" * 80)
    
    return crack_results, spalling_results

if __name__ == "__main__":
    # Train on both datasets
    train_combined()
