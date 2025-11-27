#!/usr/bin/env python3
"""
Create YOLOv11 vs Previous Training Comparison Curves
Similar to the attached image format
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for better looking plots
plt.style.use('default')
try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    print("ℹ️  Seaborn not available, using default matplotlib colors")

def load_training_data():
    """Load training data from different experiments"""
    data = {}
    
    # YOLOv11 results (new)
    yolo11_path = Path("runs/detect/yolo11_crack_detection_train/results.csv")
    if yolo11_path.exists():
        data['YOLOv11 (New)'] = pd.read_csv(yolo11_path)
        print(f"✅ Loaded YOLOv11 data: {len(data['YOLOv11 (New)'])} epochs")
    
    # Previous training results
    prev_paths = [
        ("Original Training", "runs/detect/train/results.csv"),
        ("Fixed Training", "runs/detect/crack_detection_fixed/results.csv"),
        ("Train V3", "runs/detect/train3/results.csv"),
        ("Segmentation Model", "segmentation_model/results.csv")
    ]
    
    for name, path in prev_paths:
        path = Path(path)
        if path.exists():
            try:
                df = pd.read_csv(path)
                data[name] = df
                print(f"✅ Loaded {name}: {len(df)} epochs")
            except Exception as e:
                print(f"⚠️  Could not load {name}: {e}")
    
    return data

def create_comparison_plots(data):
    """Create comparison plots similar to the attached image"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Define colors for each training run
    colors = {
        'YOLOv11 (New)': '#FF6B6B',  # Red
        'Original Training': '#4ECDC4',  # Teal
        'Fixed Training': '#45B7D1',  # Blue
        'Train V3': '#96CEB4',  # Green
        'Segmentation Model': '#FFEAA7'  # Yellow
    }
    
    # 1. Total Combined Loss (top panel)
    ax1 = plt.subplot(2, 2, (1, 2))
    ax1.set_title('Total Combined Loss', fontsize=14, fontweight='bold')
    
    for name, df in data.items():
        if len(df) > 0:
            # Calculate total loss (approximate)
            if 'train/box_loss' in df.columns and 'train/cls_loss' in df.columns:
                total_loss = df['train/box_loss'] + df['train/cls_loss']
                if 'train/dfl_loss' in df.columns:
                    total_loss += df['train/dfl_loss']
            else:
                # Fallback for different column names
                loss_cols = [col for col in df.columns if 'loss' in col.lower() and 'train' in col.lower()]
                if loss_cols:
                    total_loss = df[loss_cols].sum(axis=1)
                else:
                    continue
            
            epochs = range(1, len(total_loss) + 1)
            ax1.plot(epochs, total_loss, label=name, color=colors.get(name, 'gray'), linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, None)
    
    # 2. Box Loss (bottom left)
    ax2 = plt.subplot(2, 3, 4)
    ax2.set_title('Box Loss', fontsize=12, fontweight='bold')
    
    for name, df in data.items():
        if len(df) > 0:
            box_loss_col = None
            for col in df.columns:
                if 'box' in col.lower() and 'loss' in col.lower() and 'train' in col.lower():
                    box_loss_col = col
                    break
            
            if box_loss_col:
                epochs = range(1, len(df) + 1)
                ax2.plot(epochs, df[box_loss_col], label=name, color=colors.get(name, 'gray'), linewidth=1.5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Classification Loss (bottom middle)
    ax3 = plt.subplot(2, 3, 5)
    ax3.set_title('Classification Loss', fontsize=12, fontweight='bold')
    
    for name, df in data.items():
        if len(df) > 0:
            cls_loss_col = None
            for col in df.columns:
                if 'cls' in col.lower() and 'loss' in col.lower() and 'train' in col.lower():
                    cls_loss_col = col
                    break
            
            if cls_loss_col:
                epochs = range(1, len(df) + 1)
                ax3.plot(epochs, df[cls_loss_col], label=name, color=colors.get(name, 'gray'), linewidth=1.5)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. DFL Loss (bottom right)
    ax4 = plt.subplot(2, 3, 6)
    ax4.set_title('DFL Loss', fontsize=12, fontweight='bold')
    
    for name, df in data.items():
        if len(df) > 0:
            dfl_loss_col = None
            for col in df.columns:
                if 'dfl' in col.lower() and 'loss' in col.lower() and 'train' in col.lower():
                    dfl_loss_col = col
                    break
            
            if dfl_loss_col:
                epochs = range(1, len(df) + 1)
                ax4.plot(epochs, df[dfl_loss_col], label=name, color=colors.get(name, 'gray'), linewidth=1.5)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_performance_comparison(data):
    """Create performance metrics comparison"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {
        'YOLOv11 (New)': '#FF6B6B',  # Red
        'Original Training': '#4ECDC4',  # Teal
        'Fixed Training': '#45B7D1',  # Blue
        'Train V3': '#96CEB4',  # Green
        'Segmentation Model': '#FFEAA7'  # Yellow
    }
    
    # Precision
    ax1.set_title('Precision', fontsize=12, fontweight='bold')
    for name, df in data.items():
        if len(df) > 0:
            precision_col = None
            for col in df.columns:
                if 'precision' in col.lower():
                    precision_col = col
                    break
            
            if precision_col:
                epochs = range(1, len(df) + 1)
                ax1.plot(epochs, df[precision_col], label=name, color=colors.get(name, 'gray'), linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Recall
    ax2.set_title('Recall', fontsize=12, fontweight='bold')
    for name, df in data.items():
        if len(df) > 0:
            recall_col = None
            for col in df.columns:
                if 'recall' in col.lower():
                    recall_col = col
                    break
            
            if recall_col:
                epochs = range(1, len(df) + 1)
                ax2.plot(epochs, df[recall_col], label=name, color=colors.get(name, 'gray'), linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # mAP50
    ax3.set_title('mAP@0.5', fontsize=12, fontweight='bold')
    for name, df in data.items():
        if len(df) > 0:
            map50_col = None
            for col in df.columns:
                if 'map50' in col.lower() or ('map' in col.lower() and '50' in col):
                    map50_col = col
                    break
            
            if map50_col:
                epochs = range(1, len(df) + 1)
                ax3.plot(epochs, df[map50_col], label=name, color=colors.get(name, 'gray'), linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mAP@0.5')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # mAP50-95
    ax4.set_title('mAP@0.5:0.95', fontsize=12, fontweight='bold')
    for name, df in data.items():
        if len(df) > 0:
            map5095_col = None
            for col in df.columns:
                if 'map50-95' in col.lower() or 'mAP50-95' in col:
                    map5095_col = col
                    break
            
            if map5095_col:
                epochs = range(1, len(df) + 1)
                ax4.plot(epochs, df[map5095_col], label=name, color=colors.get(name, 'gray'), linewidth=2)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('mAP@0.5:0.95')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def main():
    print("🔍 Loading training data from all experiments...")
    data = load_training_data()
    
    if not data:
        print("❌ No training data found!")
        return
    
    print(f"\n📊 Creating comparison plots for {len(data)} training runs...")
    
    # Create loss comparison plots
    print("📈 Creating loss comparison curves...")
    loss_fig = create_comparison_plots(data)
    loss_fig.savefig('yolo11_vs_previous_loss_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: yolo11_vs_previous_loss_comparison.png")
    
    # Create performance comparison plots
    print("📊 Creating performance comparison curves...")
    perf_fig = create_performance_comparison(data)
    perf_fig.savefig('yolo11_vs_previous_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: yolo11_vs_previous_performance_comparison.png")
    
    # Print summary
    print("\n📋 Training Summary:")
    for name, df in data.items():
        if len(df) > 0:
            final_epoch = df.iloc[-1]
            print(f"\n{name}:")
            print(f"  Epochs: {len(df)}")
            
            # Find best metrics
            for metric in ['precision', 'recall', 'mAP50', 'mAP50-95']:
                for col in df.columns:
                    if metric.lower() in col.lower():
                        best_val = df[col].max()
                        print(f"  Best {metric}: {best_val:.3f}")
                        break

if __name__ == "__main__":
    main()
