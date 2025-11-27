"""
Monitor YOLOv8 Training Progress
Checks the training progress for both datasets
"""

import os
import time
from pathlib import Path

def check_training_progress():
    """Check and display training progress for both models"""
    
    print("=" * 80)
    print("YOLOV8 TRAINING PROGRESS MONITOR")
    print("=" * 80)
    
    # Check crack augmented training (DETECTION)
    crack_path = Path("runs/detect/yolov8_crack_detection")
    print("\n[1] Crack Augmented Dataset (DETECTION Model):")
    if crack_path.exists():
        results_file = crack_path / "results.csv"
        weights_path = crack_path / "weights"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                lines = f.readlines()
                epoch_count = len(lines) - 1  # Subtract header
                print(f"   ✓ Training in progress: {epoch_count}/100 epochs completed")
                if len(lines) > 1:
                    # Parse last line for metrics
                    last_line = lines[-1].strip().split(',')
                    if len(last_line) >= 10:
                        try:
                            epoch = last_line[0].strip()
                            box_loss = float(last_line[2])
                            cls_loss = float(last_line[3])
                            dfl_loss = float(last_line[4])
                            precision = float(last_line[5])
                            recall = float(last_line[6])
                            map50 = float(last_line[7])
                            map50_95 = float(last_line[8])
                            print(f"   📊 Latest metrics (Epoch {epoch}):")
                            print(f"      - Box Loss: {box_loss:.4f}, Cls Loss: {cls_loss:.4f}, DFL Loss: {dfl_loss:.4f}")
                            print(f"      - Precision: {precision:.4f}, Recall: {recall:.4f}")
                            print(f"      - mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")
                        except:
                            print(f"   Latest line: {last_line[:5]}...")
        else:
            print("   ⏳ Training in progress... (generating results.csv)")
        
        if weights_path.exists():
            weights = list(weights_path.glob("*.pt"))
            print(f"   📦 Weights saved: {len(weights)} checkpoint(s)")
            for w in weights:
                size_mb = w.stat().st_size / (1024 * 1024)
                mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(w.stat().st_mtime))
                print(f"      - {w.name} ({size_mb:.2f} MB) - {mtime}")
    else:
        print("   ⏸️  Not started yet")
    
    # Check spalling rebar training (SEGMENTATION)
    spalling_path = Path("runs/segment/yolov8_spalling_rebar_segmentation")
    print("\n[2] Spalling & Exposed Rebar Dataset (SEGMENTATION Model):")
    if spalling_path.exists():
        results_file = spalling_path / "results.csv"
        weights_path = spalling_path / "weights"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                lines = f.readlines()
                epoch_count = len(lines) - 1
                print(f"   ✓ Training in progress: {epoch_count}/100 epochs completed")
                if len(lines) > 1:
                    # Parse last line for metrics
                    last_line = lines[-1].strip().split(',')
                    if len(last_line) >= 15:
                        try:
                            epoch = last_line[0].strip()
                            box_loss = float(last_line[2])
                            seg_loss = float(last_line[3])
                            cls_loss = float(last_line[4])
                            print(f"   📊 Latest metrics (Epoch {epoch}):")
                            print(f"      - Box Loss: {box_loss:.4f}, Seg Loss: {seg_loss:.4f}, Cls Loss: {cls_loss:.4f}")
                        except:
                            print(f"   Latest line: {last_line[:5]}...")
        else:
            print("   ⏳ Training in progress... (generating results.csv)")
        
        if weights_path.exists():
            weights = list(weights_path.glob("*.pt"))
            print(f"   📦 Weights saved: {len(weights)} checkpoint(s)")
            for w in weights:
                size_mb = w.stat().st_size / (1024 * 1024)
                mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(w.stat().st_mtime))
                print(f"      - {w.name} ({size_mb:.2f} MB) - {mtime}")
    else:
        print("   ⏸️  Not started yet (waiting for crack training to complete)")
    
    print("\n" + "=" * 80)
    print("ℹ️  Note: Training on CPU may take several hours per dataset")
    print("   You can monitor progress by running this script periodically")
    print("=" * 80)

if __name__ == "__main__":
    os.chdir("/Users/prashant/Documents/coding/demo")
    check_training_progress()
