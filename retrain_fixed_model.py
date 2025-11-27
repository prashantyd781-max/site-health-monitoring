#!/usr/bin/env python3
"""
Retrain the model with the fixed dataset (non-interactive version)
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

def retrain_model():
    """Retrain the model with corrected dataset"""
    
    print("🚀 RETRAINING MODEL WITH FIXED DATASET")
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
    
    # Improved training parameters based on best practices
    training_params = {
        'data': str(data_yaml),
        'epochs': 150,           # More epochs for better convergence
        'imgsz': 640,           # Standard size
        'batch': 8,             # Smaller batch for stability
        'lr0': 0.001,           # Lower learning rate for stable training
        'weight_decay': 0.0005,
        'warmup_epochs': 5,     # More warmup
        'patience': 40,         # More patience
        'save_period': 25,      # Save every 25 epochs
        'device': device,
        'workers': 2,           # Fewer workers
        'project': 'runs/detect',
        'name': 'crack_detection_fixed',
        'exist_ok': True,
        'verbose': True,
        'plots': True,
        'save': True,
        'amp': False,           # Disable mixed precision for stability
        'optimizer': 'AdamW',   # Better optimizer
        'close_mosaic': 20      # Disable mosaic in last epochs
    }
    
    print("⚙️  Optimized Training Parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    try:
        print("\n🏋️‍♂️ Starting optimized training...")
        print("This will take significantly longer but should produce much better results...")
        
        results = model.train(**training_params)
        
        print("\n✅ Training completed successfully!")
        
        # Copy the best model
        best_model_path = Path("runs/detect/crack_detection_fixed/weights/best.pt")
        if best_model_path.exists():
            main_model_path = script_dir / "yolov8_crack_detection_WORKING.pt"
            shutil.copy2(best_model_path, main_model_path)
            print(f"🏆 Working model saved as: {main_model_path}")
            
            # Read and display final metrics
            results_csv = Path("runs/detect/crack_detection_fixed/results.csv")
            if results_csv.exists():
                import pandas as pd
                df = pd.read_csv(results_csv)
                final_row = df.iloc[-1]
                
                print(f"\n📊 FINAL TRAINING METRICS:")
                print(f"  mAP50:     {final_row['metrics/mAP50(B)']:.4f}")
                print(f"  mAP50-95:  {final_row['metrics/mAP50-95(B)']:.4f}") 
                print(f"  Precision: {final_row['metrics/precision(B)']:.4f}")
                print(f"  Recall:    {final_row['metrics/recall(B)']:.4f}")
                
                # Industry comparison
                print(f"\n🏭 PERFORMANCE ASSESSMENT:")
                mAP50 = final_row['metrics/mAP50(B)']
                precision = final_row['metrics/precision(B)']
                recall = final_row['metrics/recall(B)']
                
                if mAP50 > 0.5:
                    print("✅ mAP50 > 0.5: EXCELLENT performance!")
                elif mAP50 > 0.3:
                    print("🔶 mAP50 > 0.3: GOOD performance")
                elif mAP50 > 0.1:
                    print("⚠️ mAP50 > 0.1: FAIR performance")
                else:
                    print("❌ mAP50 < 0.1: POOR performance")
                
                if precision > 0.7 and recall > 0.6:
                    print("🎯 Model meets industry standards for crack detection!")
                elif precision > 0.5 and recall > 0.5:
                    print("📈 Model shows decent performance - consider more training data")
                else:
                    print("🔄 Model needs improvement - consider data augmentation")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = retrain_model()
    
    if success:
        print(f"\n🎉 RETRAINING COMPLETED SUCCESSFULLY!")
        print(f"📁 Check 'runs/detect/crack_detection_fixed' for detailed results")
        print(f"🏆 Your working model: 'yolov8_crack_detection_WORKING.pt'")
        print(f"\n📋 Next steps:")
        print(f"1. Test the new model on some images")
        print(f"2. Compare performance with the old model")
        print(f"3. Use the working model for crack detection")
    else:
        print(f"\n❌ Retraining failed. Check the error messages above.")
