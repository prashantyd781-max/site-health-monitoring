"""
Monitor Spalling & Rebar Segmentation Training Progress
Shows real-time training metrics and estimated completion time
"""

import os
import csv
from datetime import datetime, timedelta
import time

def monitor_training():
    """Monitor the training progress for Spalling & Rebar segmentation"""
    
    results_file = 'runs/segment/yolov8_spalling_rebar_final/results.csv'
    
    print("\n" + "=" * 80)
    print("🔍 MONITORING SPALLING & REBAR SEGMENTATION TRAINING")
    print("=" * 80)
    
    if not os.path.exists(results_file):
        print("\n⚠️  Training not started yet or results file not found!")
        print(f"   Expected location: {results_file}")
        print("\n💡 The training takes a few minutes to initialize.")
        print("   Please wait and run this script again in 2-3 minutes.")
        return
    
    # Read the CSV file
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("\n⚠️  No training data available yet!")
        return
    
    # Get the latest epoch
    latest = rows[-1]
    total_epochs = 100
    current_epoch = int(latest['                  epoch'].strip())
    
    print(f"\n📊 TRAINING PROGRESS:")
    print(f"   Epochs: {current_epoch}/{total_epochs} ({current_epoch}%)")
    print(f"   Progress: {'█' * (current_epoch // 2)}{'░' * ((total_epochs - current_epoch) // 2)}")
    
    # Calculate estimated time remaining
    if len(rows) >= 2:
        # Estimate based on recent epochs
        recent_epochs = min(5, len(rows))
        start_time = datetime.now() - timedelta(minutes=recent_epochs * 1.5)  # rough estimate
        time_per_epoch = 1.5  # minutes per epoch on GPU
        remaining_epochs = total_epochs - current_epoch
        eta_minutes = remaining_epochs * time_per_epoch
        eta = datetime.now() + timedelta(minutes=eta_minutes)
        
        print(f"\n⏱️  ESTIMATED TIME:")
        print(f"   Time per epoch: ~{time_per_epoch:.1f} minutes")
        print(f"   Remaining epochs: {remaining_epochs}")
        print(f"   Estimated completion: ~{eta_minutes/60:.1f} hours")
        print(f"   ETA: {eta.strftime('%I:%M %p, %b %d')}")
    
    # Latest metrics
    print(f"\n📈 LATEST METRICS (Epoch {current_epoch}):")
    print(f"   Box Loss:      {float(latest['         train/box_loss'].strip()):.4f}")
    print(f"   Seg Loss:      {float(latest['         train/seg_loss'].strip()):.4f}")
    print(f"   Class Loss:    {float(latest['         train/cls_loss'].strip()):.4f}")
    print(f"   DFL Loss:      {float(latest['         train/dfl_loss'].strip()):.4f}")
    
    if '           metrics/mAP50(B)' in latest:
        print(f"\n🎯 VALIDATION METRICS:")
        print(f"   Box Precision:   {float(latest['      metrics/precision(B)'].strip()):.4f}")
        print(f"   Box Recall:      {float(latest['         metrics/recall(B)'].strip()):.4f}")
        print(f"   Box mAP50:       {float(latest['           metrics/mAP50(B)'].strip()):.4f}")
        print(f"   Box mAP50-95:    {float(latest['        metrics/mAP50-95(B)'].strip()):.4f}")
        print(f"   Mask Precision:  {float(latest['      metrics/precision(M)'].strip()):.4f}")
        print(f"   Mask Recall:     {float(latest['         metrics/recall(M)'].strip()):.4f}")
        print(f"   Mask mAP50:      {float(latest['           metrics/mAP50(M)'].strip()):.4f}")
        print(f"   Mask mAP50-95:   {float(latest['        metrics/mAP50-95(M)'].strip()):.4f}")
    
    # Show progress trend
    if len(rows) >= 5:
        print(f"\n📉 LOSS TREND (Last 5 Epochs):")
        for i in range(-5, 0):
            epoch_num = int(rows[i]['                  epoch'].strip())
            box_loss = float(rows[i]['         train/box_loss'].strip())
            seg_loss = float(rows[i]['         train/seg_loss'].strip())
            print(f"   Epoch {epoch_num:2d}: Box={box_loss:.4f}, Seg={seg_loss:.4f}")
    
    print(f"\n💾 MODEL LOCATION:")
    print(f"   Best model: runs/segment/yolov8_spalling_rebar_final/weights/best.pt")
    print(f"   Last model: runs/segment/yolov8_spalling_rebar_final/weights/last.pt")
    
    print(f"\n📝 LOG FILE:")
    print(f"   Live updates: tail -f spalling_training.log")
    
    print("\n" + "=" * 80)
    print("💡 TIP: Run this script periodically to check progress")
    print("   Or use: tail -f spalling_training.log for live updates")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        monitor_training()
    except Exception as e:
        print(f"\n❌ Error monitoring training: {e}")
        print("\n💡 Make sure training has started and generated results.csv")


