#!/usr/bin/env python3
"""
Dataset Fix and Retraining Script
This script identifies and fixes dataset issues, then retrains the model properly.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

def analyze_labels(labels_dir):
    """Analyze label files to identify format issues"""
    
    print(f"🔍 Analyzing labels in: {labels_dir}")
    
    label_files = list(Path(labels_dir).glob("*.txt"))
    issues = {
        'empty_files': [],
        'segmentation_format': [],
        'invalid_format': [],
        'valid_files': [],
        'total_annotations': 0
    }
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        if not lines or all(not line.strip() for line in lines):
            issues['empty_files'].append(label_file.name)
            continue
            
        file_valid = True
        file_annotations = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            
            # Check if this is segmentation format (more than 5 values)
            if len(parts) > 5:
                issues['segmentation_format'].append(label_file.name)
                file_valid = False
                break
            # Check if this is valid bounding box format (exactly 5 values)
            elif len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check if coordinates are normalized (0-1 range)
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        issues['invalid_format'].append(f"{label_file.name}: coords out of range")
                        file_valid = False
                        break
                        
                    file_annotations += 1
                    
                except ValueError:
                    issues['invalid_format'].append(f"{label_file.name}: non-numeric values")
                    file_valid = False
                    break
            else:
                issues['invalid_format'].append(f"{label_file.name}: wrong number of values ({len(parts)})")
                file_valid = False
                break
        
        if file_valid:
            issues['valid_files'].append(label_file.name)
            issues['total_annotations'] += file_annotations
    
    return issues

def fix_segmentation_labels(labels_dir, backup_dir="labels_backup"):
    """Convert segmentation format labels to bounding box format"""
    
    print(f"🛠️  Fixing segmentation format labels...")
    
    # Create backup
    backup_path = Path(labels_dir).parent / backup_dir
    if backup_path.exists():
        shutil.rmtree(backup_path)
    shutil.copytree(labels_dir, backup_path)
    print(f"📁 Backup created: {backup_path}")
    
    issues = analyze_labels(labels_dir)
    fixed_count = 0
    
    for seg_file in issues['segmentation_format']:
        label_path = Path(labels_dir) / seg_file
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) > 5:  # Segmentation format
                class_id = parts[0]
                # Extract x,y coordinates (skip class_id)
                coords = [float(x) for x in parts[1:]]
                
                # Convert polygon to bounding box
                x_coords = coords[0::2]  # Every even index (x coordinates)
                y_coords = coords[1::2]  # Every odd index (y coordinates)
                
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                
                # Convert to YOLO bounding box format
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                # Ensure coordinates are in valid range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                fixed_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                fixed_lines.append(fixed_line)
                
            else:  # Already in bounding box format
                fixed_lines.append(line)
        
        # Write fixed file
        with open(label_path, 'w') as f:
            f.write('\n'.join(fixed_lines) + '\n')
        
        fixed_count += 1
    
    print(f"✅ Fixed {fixed_count} segmentation format files")
    return fixed_count

def retrain_model_properly():
    """Retrain the model with corrected dataset and proper parameters"""
    
    print("\n🚀 RETRAINING MODEL WITH FIXED DATASET")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    
    # Paths
    script_dir = Path(__file__).parent
    data_yaml = script_dir / "model_data" / "data.yaml"
    base_model = script_dir / "yolo11n.pt"
    
    # Load fresh model
    model = YOLO(str(base_model))
    
    # Improved training parameters
    training_params = {
        'data': str(data_yaml),
        'epochs': 150,           # Increased epochs
        'imgsz': 640,           # Standard image size
        'batch': 8,             # Smaller batch for stability
        'lr0': 0.001,           # Lower initial learning rate
        'weight_decay': 0.0005,
        'warmup_epochs': 5,     # More warmup epochs
        'patience': 30,         # More patience for early stopping
        'save_period': 20,      # Save less frequently
        'device': device,
        'workers': 2,           # Fewer workers for stability
        'project': 'runs/detect',
        'name': 'crack_detection_fixed',
        'exist_ok': True,
        'verbose': True,
        'plots': True,
        'save': True
    }
    
    print("⚙️  Improved Training Parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    try:
        print("\n🏋️‍♂️ Starting improved training...")
        results = model.train(**training_params)
        
        print("\n✅ Training completed successfully!")
        
        # Copy the best model
        best_model_path = Path("runs/detect/crack_detection_fixed/weights/best.pt")
        if best_model_path.exists():
            main_model_path = script_dir / "yolov8_crack_detection_fixed.pt"
            shutil.copy2(best_model_path, main_model_path)
            print(f"🏆 Fixed model saved: {main_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

def main():
    script_dir = Path(__file__).parent
    
    print("🔧 DATASET ANALYSIS AND FIXING")
    print("=" * 50)
    
    # Analyze all label directories
    for split in ['train', 'valid', 'test']:
        labels_dir = script_dir / "model_data" / split / "labels"
        if labels_dir.exists():
            print(f"\n📊 {split.upper()} SET ANALYSIS:")
            issues = analyze_labels(labels_dir)
            
            print(f"  📁 Total files: {len(list(labels_dir.glob('*.txt')))}")
            print(f"  ✅ Valid files: {len(issues['valid_files'])}")
            print(f"  📦 Total annotations: {issues['total_annotations']}")
            print(f"  🔄 Segmentation format: {len(issues['segmentation_format'])}")
            print(f"  ❌ Invalid format: {len(issues['invalid_format'])}")
            print(f"  📄 Empty files: {len(issues['empty_files'])}")
            
            if issues['segmentation_format']:
                print(f"  🛠️  Fixing {len(issues['segmentation_format'])} segmentation files...")
                fix_segmentation_labels(labels_dir)
            
            if issues['invalid_format']:
                print("  ⚠️  Invalid format issues found:")
                for issue in issues['invalid_format'][:5]:  # Show first 5
                    print(f"    - {issue}")
    
    # Re-analyze after fixes
    print(f"\n✅ DATASET FIXED - ANALYSIS AFTER FIXES:")
    total_annotations = 0
    for split in ['train', 'valid', 'test']:
        labels_dir = script_dir / "model_data" / split / "labels"
        if labels_dir.exists():
            issues = analyze_labels(labels_dir)
            print(f"  {split}: {issues['total_annotations']} annotations in {len(issues['valid_files'])} files")
            total_annotations += issues['total_annotations']
    
    print(f"📊 TOTAL ANNOTATIONS: {total_annotations}")
    
    if total_annotations > 0:
        print(f"\n🚀 Dataset is ready for training!")
        
        # Ask user if they want to retrain
        print(f"\n🤔 Do you want to retrain the model now with fixed dataset?")
        user_choice = input("Enter 'y' to continue or any other key to skip: ").lower().strip()
        
        if user_choice == 'y':
            success = retrain_model_properly()
            if success:
                print(f"\n🎉 MODEL RETRAINING COMPLETED!")
                print(f"📁 Check 'runs/detect/crack_detection_fixed' for results")
                print(f"🏆 New model: 'yolov8_crack_detection_fixed.pt'")
            else:
                print(f"\n❌ Retraining failed. Please check the error messages.")
        else:
            print(f"\n📝 You can retrain later by running this script again.")
    else:
        print(f"\n❌ No valid annotations found. Please check your dataset.")

if __name__ == "__main__":
    main()
