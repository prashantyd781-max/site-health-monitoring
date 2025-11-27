#!/usr/bin/env python3
"""
YOLOv11 Crack Detection Training Script
This script trains a YOLOv11 model on the crack detection dataset.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    print("🚀 Starting YOLOv11 Crack Detection Training")
    print("=" * 50)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    
    # Paths
    script_dir = Path(__file__).parent
    data_yaml = script_dir / "model_data" / "data.yaml"
    base_model = script_dir / "yolo11n.pt"
    
    # Verify required files exist
    if not data_yaml.exists():
        print(f"❌ Error: Dataset config file not found at {data_yaml}")
        sys.exit(1)
        
    if not base_model.exists():
        print(f"❌ Error: Base model not found at {base_model}")
        sys.exit(1)
    
    print(f"📁 Dataset config: {data_yaml}")
    print(f"🤖 Base model: {base_model}")
    
    # Load the YOLOv11 model
    print("\n🔄 Loading YOLOv11 model...")
    model = YOLO(str(base_model))
    
    # Training parameters
    training_params = {
        'data': str(data_yaml),
        'epochs': 100,           # Number of training epochs
        'imgsz': 640,           # Image size for training
        'batch': 16,            # Batch size (adjust based on GPU memory)
        'lr0': 0.01,            # Initial learning rate
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 3,      # Warmup epochs
        'patience': 50,          # Early stopping patience
        'save_period': 10,       # Save model every N epochs
        'device': device,        # Use GPU if available
        'workers': 4,            # Number of data loading workers
        'project': 'runs/detect', # Project directory
        'name': 'yolo11_crack_detection_train', # Experiment name
        'exist_ok': True,        # Overwrite existing experiment
    }
    
    print("\n⚙️  Training Parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    # Start training
    print("\n🏋️‍♂️ Starting training...")
    print("This may take a while depending on your hardware...")
    
    try:
        results = model.train(**training_params)
        
        print("\n✅ Training completed successfully!")
        print(f"📊 Results saved in: runs/detect/yolo11_crack_detection_train")
        
        # Find the best model
        best_model_path = Path("runs/detect/yolo11_crack_detection_train/weights/best.pt")
        if best_model_path.exists():
            print(f"🏆 Best model saved at: {best_model_path}")
            
            # Copy the best model to the main directory for easy access
            import shutil
            main_model_path = script_dir / "yolo11_crack_detection_best.pt"
            shutil.copy2(best_model_path, main_model_path)
            print(f"📋 Model copied to: {main_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        return False

def validate_dataset():
    """Validate the dataset structure before training"""
    print("\n🔍 Validating dataset structure...")
    
    script_dir = Path(__file__).parent
    model_data_dir = script_dir / "model_data"
    
    required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels", "test/images", "test/labels"]
    
    for dir_path in required_dirs:
        full_path = model_data_dir / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {full_path}")
            return False
        
        # Count files
        if "images" in dir_path:
            image_count = len(list(full_path.glob("*.jpg"))) + len(list(full_path.glob("*.png")))
            print(f"📸 {dir_path}: {image_count} images")
        else:
            label_count = len(list(full_path.glob("*.txt")))
            print(f"🏷️  {dir_path}: {label_count} labels")
    
    print("✅ Dataset structure is valid!")
    return True

if __name__ == "__main__":
    print("🔍 Validating dataset before training...")
    if not validate_dataset():
        print("❌ Dataset validation failed. Please check your dataset structure.")
        sys.exit(1)
    
    print("\n🚀 Starting training process...")
    success = main()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Check the 'runs/detect/yolo11_crack_detection_train' directory for training results")
        print("🏆 Best model available as 'yolo11_crack_detection_best.pt'")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)





