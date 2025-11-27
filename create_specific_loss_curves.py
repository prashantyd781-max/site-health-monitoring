#!/usr/bin/env python3
"""
Create Specific Loss Curves: Box Loss, Classification Loss, DFL Loss, Total Combined Loss
Similar to the attached image format
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_all_training_data():
    """Load training data from all experiments"""
    data = {}
    
    # Define all possible training runs with their paths and display names
    training_runs = [
        ("YOLOv11 (New)", "runs/detect/yolo11_crack_detection_train/results.csv", '#FF6B6B'),
        ("Original Training (Train)", "runs/detect/train/results.csv", '#4ECDC4'),
        ("Fixed Training (Train)", "runs/detect/crack_detection_fixed/results.csv", '#45B7D1'),
        ("Train V3 (Train)", "runs/detect/train3/results.csv", '#96CEB4'),
        ("Segmentation Model (Train)", "segmentation_model/results.csv", '#FFEAA7'),
    ]
    
    for name, path, color in training_runs:
        path = Path(path)
        if path.exists():
            try:
                df = pd.read_csv(path)
                data[name] = {'data': df, 'color': color}
                print(f"✅ Loaded {name}: {len(df)} epochs")
            except Exception as e:
                print(f"⚠️  Could not load {name}: {e}")
    
    return data

def create_loss_curves():
    """Create the 4 specific loss curves"""
    
    # Load data
    print("🔍 Loading training data from all experiments...")
    all_data = load_all_training_data()
    
    if not all_data:
        print("❌ No training data found!")
        return
    
    # Create figure with 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Loss Curves Comparison', fontsize=16, fontweight='bold')
    
    # 1. Total Combined Loss (top-left)
    ax1.set_title('Total Combined Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    for name, info in all_data.items():
        df = info['data']
        color = info['color']
        
        if len(df) > 0:
            # Calculate total training loss
            total_train_loss = None
            total_val_loss = None
            
            # Training loss
            if 'train/box_loss' in df.columns and 'train/cls_loss' in df.columns:
                total_train_loss = df['train/box_loss'] + df['train/cls_loss']
                if 'train/dfl_loss' in df.columns:
                    total_train_loss += df['train/dfl_loss']
            
            # Validation loss
            if 'val/box_loss' in df.columns and 'val/cls_loss' in df.columns:
                total_val_loss = df['val/box_loss'] + df['val/cls_loss'] 
                if 'val/dfl_loss' in df.columns:
                    total_val_loss += df['val/dfl_loss']
            
            epochs = range(1, len(df) + 1)
            
            if total_train_loss is not None:
                ax1.plot(epochs, total_train_loss, label=f'{name} (Train)', 
                        color=color, linewidth=2, linestyle='-')
            
            if total_val_loss is not None:
                ax1.plot(epochs, total_val_loss, label=f'{name} (Val)', 
                        color=color, linewidth=2, linestyle='--')
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.set_ylim(0, None)
    
    # 2. Box Loss (top-right)
    ax2.set_title('Box Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    for name, info in all_data.items():
        df = info['data']
        color = info['color']
        
        if len(df) > 0:
            epochs = range(1, len(df) + 1)
            
            # Training box loss
            if 'train/box_loss' in df.columns:
                ax2.plot(epochs, df['train/box_loss'], label=f'{name} (Train)', 
                        color=color, linewidth=2, linestyle='-')
            
            # Validation box loss
            if 'val/box_loss' in df.columns:
                ax2.plot(epochs, df['val/box_loss'], label=f'{name} (Val)', 
                        color=color, linewidth=2, linestyle='--')
    
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, None)
    
    # 3. Classification Loss (bottom-left)
    ax3.set_title('Classification Loss', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    for name, info in all_data.items():
        df = info['data']
        color = info['color']
        
        if len(df) > 0:
            epochs = range(1, len(df) + 1)
            
            # Training classification loss
            if 'train/cls_loss' in df.columns:
                ax3.plot(epochs, df['train/cls_loss'], label=f'{name} (Train)', 
                        color=color, linewidth=2, linestyle='-')
            
            # Validation classification loss
            if 'val/cls_loss' in df.columns:
                ax3.plot(epochs, df['val/cls_loss'], label=f'{name} (Val)', 
                        color=color, linewidth=2, linestyle='--')
    
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, None)
    
    # 4. DFL Loss (bottom-right)
    ax4.set_title('DFL Loss', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.grid(True, alpha=0.3)
    
    for name, info in all_data.items():
        df = info['data']
        color = info['color']
        
        if len(df) > 0:
            epochs = range(1, len(df) + 1)
            
            # Training DFL loss
            if 'train/dfl_loss' in df.columns:
                ax4.plot(epochs, df['train/dfl_loss'], label=f'{name} (Train)', 
                        color=color, linewidth=2, linestyle='-')
            
            # Validation DFL loss
            if 'val/dfl_loss' in df.columns:
                ax4.plot(epochs, df['val/dfl_loss'], label=f'{name} (Val)', 
                        color=color, linewidth=2, linestyle='--')
    
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, None)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'yolo11_detailed_loss_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved detailed loss curves: {output_file}")
    
    return fig

def create_yolo11_only_curves():
    """Create loss curves for YOLOv11 only (cleaner view)"""
    
    # Load YOLOv11 data
    yolo11_path = Path("runs/detect/yolo11_crack_detection_train/results.csv")
    if not yolo11_path.exists():
        print("❌ YOLOv11 results not found!")
        return
    
    df = pd.read_csv(yolo11_path)
    epochs = range(1, len(df) + 1)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLOv11 Training Loss Curves', fontsize=16, fontweight='bold')
    
    colors = {'train': '#FF6B6B', 'val': '#4ECDC4'}
    
    # 1. Total Combined Loss
    ax1.set_title('Total Combined Loss', fontsize=14, fontweight='bold')
    total_train = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    total_val = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
    
    ax1.plot(epochs, total_train, label='Training', color=colors['train'], linewidth=2)
    ax1.plot(epochs, total_val, label='Validation', color=colors['val'], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box Loss
    ax2.set_title('Box Loss', fontsize=14, fontweight='bold')
    ax2.plot(epochs, df['train/box_loss'], label='Training', color=colors['train'], linewidth=2)
    ax2.plot(epochs, df['val/box_loss'], label='Validation', color=colors['val'], linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Classification Loss
    ax3.set_title('Classification Loss', fontsize=14, fontweight='bold')
    ax3.plot(epochs, df['train/cls_loss'], label='Training', color=colors['train'], linewidth=2)
    ax3.plot(epochs, df['val/cls_loss'], label='Validation', color=colors['val'], linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. DFL Loss
    ax4.set_title('DFL Loss', fontsize=14, fontweight='bold')
    ax4.plot(epochs, df['train/dfl_loss'], label='Training', color=colors['train'], linewidth=2)
    ax4.plot(epochs, df['val/dfl_loss'], label='Validation', color=colors['val'], linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'yolo11_only_loss_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved YOLOv11-only loss curves: {output_file}")
    
    return fig

def print_loss_summary():
    """Print summary of final loss values"""
    yolo11_path = Path("runs/detect/yolo11_crack_detection_train/results.csv")
    if not yolo11_path.exists():
        return
    
    df = pd.read_csv(yolo11_path)
    final_epoch = df.iloc[-1]
    
    print("\n📊 YOLOv11 Final Loss Values (Epoch 100):")
    print(f"  Training Losses:")
    print(f"    Box Loss: {final_epoch['train/box_loss']:.4f}")
    print(f"    Classification Loss: {final_epoch['train/cls_loss']:.4f}")
    print(f"    DFL Loss: {final_epoch['train/dfl_loss']:.4f}")
    
    total_train = final_epoch['train/box_loss'] + final_epoch['train/cls_loss'] + final_epoch['train/dfl_loss']
    print(f"    Total Combined: {total_train:.4f}")
    
    print(f"  Validation Losses:")
    print(f"    Box Loss: {final_epoch['val/box_loss']:.4f}")
    print(f"    Classification Loss: {final_epoch['val/cls_loss']:.4f}")
    print(f"    DFL Loss: {final_epoch['val/dfl_loss']:.4f}")
    
    total_val = final_epoch['val/box_loss'] + final_epoch['val/cls_loss'] + final_epoch['val/dfl_loss']
    print(f"    Total Combined: {total_val:.4f}")

def main():
    print("📊 Creating specific loss curves...")
    
    # Create comparison curves with all training runs
    create_loss_curves()
    
    # Create YOLOv11-only curves for cleaner view
    create_yolo11_only_curves()
    
    # Print summary
    print_loss_summary()
    
    print("\n🎯 Generated Files:")
    print("  📈 yolo11_detailed_loss_curves.png - All training runs comparison")
    print("  📊 yolo11_only_loss_curves.png - YOLOv11 training & validation only")

if __name__ == "__main__":
    main()





