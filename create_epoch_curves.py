#!/usr/bin/env python3
"""
Comprehensive Epoch Curves Visualization
Creates detailed training progression curves for YOLO crack detection models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_training_data():
    """Load all training results data"""
    data = {}
    
    # Original training (with dataset issues)
    original_path = Path("runs/detect/crack_detection_train/results.csv")
    if original_path.exists():
        data['Original Training'] = pd.read_csv(original_path)
        print(f"✅ Loaded Original Training: {len(data['Original Training'])} epochs")
    
    # Fixed training
    fixed_path = Path("runs/detect/crack_detection_fixed/results.csv")
    if fixed_path.exists():
        data['Fixed Training'] = pd.read_csv(fixed_path)
        print(f"✅ Loaded Fixed Training: {len(data['Fixed Training'])} epochs")
    
    # Segmentation model
    seg_path = Path("segmentation_model/results.csv")
    if seg_path.exists():
        data['Segmentation Model'] = pd.read_csv(seg_path)
        print(f"✅ Loaded Segmentation Model: {len(data['Segmentation Model'])} epochs")
    
    return data

def create_loss_curves(data):
    """Create loss progression curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('📉 Training & Validation Loss Curves', fontsize=16, fontweight='bold')
    
    loss_types = [
        ('train/box_loss', 'val/box_loss', 'Box Loss'),
        ('train/cls_loss', 'val/cls_loss', 'Classification Loss'),
        ('train/dfl_loss', 'val/dfl_loss', 'DFL Loss')
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for idx, (train_col, val_col, title) in enumerate(loss_types):
        ax = axes[0, idx]
        color_idx = 0
        
        for name, df in data.items():
            if train_col in df.columns and val_col in df.columns:
                epochs = range(1, len(df) + 1)
                
                # Training loss
                ax.plot(epochs, df[train_col], 
                       label=f'{name} (Train)', 
                       color=colors[color_idx], 
                       linewidth=2, alpha=0.8)
                
                # Validation loss
                ax.plot(epochs, df[val_col], 
                       label=f'{name} (Val)', 
                       color=colors[color_idx], 
                       linestyle='--', linewidth=2, alpha=0.8)
                
                color_idx += 2
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Combined loss plot
    ax = axes[1, :]
    ax = plt.subplot(2, 1, 2)
    
    for name, df in data.items():
        if all(col in df.columns for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
            epochs = range(1, len(df) + 1)
            total_train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
            total_val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
            
            ax.plot(epochs, total_train_loss, label=f'{name} (Train Total)', linewidth=2)
            ax.plot(epochs, total_val_loss, label=f'{name} (Val Total)', linestyle='--', linewidth=2)
    
    ax.set_title('Total Combined Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('epoch_curves_losses.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: epoch_curves_losses.png")
    return fig

def create_performance_metrics(data):
    """Create performance metrics curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📊 Performance Metrics Over Epochs', fontsize=16, fontweight='bold')
    
    metrics = [
        ('metrics/precision(B)', 'Precision', '🎯'),
        ('metrics/recall(B)', 'Recall', '🔍'),
        ('metrics/mAP50(B)', 'mAP@0.5', '🏆'),
        ('metrics/mAP50-95(B)', 'mAP@0.5:0.95', '⭐')
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for idx, (metric, title, emoji) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        color_idx = 0
        
        for name, df in data.items():
            if metric in df.columns:
                epochs = range(1, len(df) + 1)
                values = df[metric]
                
                ax.plot(epochs, values, 
                       label=name, 
                       color=colors[color_idx], 
                       linewidth=3, alpha=0.8,
                       marker='o' if len(epochs) < 50 else None,
                       markersize=3)
                
                # Add final value annotation
                final_val = values.iloc[-1]
                ax.annotate(f'{final_val:.3f}', 
                           xy=(len(epochs), final_val),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
                
                color_idx += 1
        
        ax.set_title(f'{emoji} {title}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
    
    plt.tight_layout()
    plt.savefig('epoch_curves_metrics.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: epoch_curves_metrics.png")
    return fig

def create_learning_rate_curves(data):
    """Create learning rate progression curves"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('📈 Learning Rate Schedule', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    color_idx = 0
    
    for name, df in data.items():
        if 'lr/pg0' in df.columns:
            epochs = range(1, len(df) + 1)
            lr = df['lr/pg0']
            
            ax.plot(epochs, lr, 
                   label=name, 
                   color=colors[color_idx], 
                   linewidth=2, alpha=0.8)
            
            color_idx += 1
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('epoch_curves_learning_rate.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: epoch_curves_learning_rate.png")
    return fig

def create_segmentation_comparison(data):
    """Create segmentation-specific plots if available"""
    seg_data = None
    for name, df in data.items():
        if 'metrics/precision(M)' in df.columns:  # Has mask metrics
            seg_data = (name, df)
            break
    
    if not seg_data:
        print("ℹ️ No segmentation data found, skipping segmentation plots")
        return None
    
    name, df = seg_data
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'🎭 Segmentation Model: Detection vs Mask Performance', 
                 fontsize=16, fontweight='bold')
    
    epochs = range(1, len(df) + 1)
    
    # Precision comparison
    ax = axes[0, 0]
    ax.plot(epochs, df['metrics/precision(B)'], label='Detection (Box)', 
           linewidth=3, color='#FF6B6B')
    ax.plot(epochs, df['metrics/precision(M)'], label='Segmentation (Mask)', 
           linewidth=3, color='#4ECDC4')
    ax.set_title('🎯 Precision: Detection vs Segmentation')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Recall comparison
    ax = axes[0, 1]
    ax.plot(epochs, df['metrics/recall(B)'], label='Detection (Box)', 
           linewidth=3, color='#FF6B6B')
    ax.plot(epochs, df['metrics/recall(M)'], label='Segmentation (Mask)', 
           linewidth=3, color='#4ECDC4')
    ax.set_title('🔍 Recall: Detection vs Segmentation')
    ax.set_ylabel('Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # mAP@0.5 comparison
    ax = axes[1, 0]
    ax.plot(epochs, df['metrics/mAP50(B)'], label='Detection (Box)', 
           linewidth=3, color='#FF6B6B')
    ax.plot(epochs, df['metrics/mAP50(M)'], label='Segmentation (Mask)', 
           linewidth=3, color='#4ECDC4')
    ax.set_title('🏆 mAP@0.5: Detection vs Segmentation')
    ax.set_ylabel('mAP@0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Combined loss for segmentation
    ax = axes[1, 1]
    if 'train/seg_loss' in df.columns:
        ax.plot(epochs, df['train/box_loss'], label='Box Loss', linewidth=2)
        ax.plot(epochs, df['train/seg_loss'], label='Segmentation Loss', linewidth=2)
        ax.plot(epochs, df['train/cls_loss'], label='Classification Loss', linewidth=2)
        ax.plot(epochs, df['train/dfl_loss'], label='DFL Loss', linewidth=2)
    
    ax.set_title('📉 Training Loss Components')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('epoch_curves_segmentation.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: epoch_curves_segmentation.png")
    return fig

def create_training_summary(data):
    """Create a comprehensive training summary"""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('📋 Training Performance Summary', fontsize=16, fontweight='bold')
    
    # Create summary table
    summary_data = []
    
    for name, df in data.items():
        if len(df) > 0:
            final_epoch = df.iloc[-1]
            best_map = df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else 0
            best_precision = df['metrics/precision(B)'].max() if 'metrics/precision(B)' in df.columns else 0
            best_recall = df['metrics/recall(B)'].max() if 'metrics/recall(B)' in df.columns else 0
            
            summary_data.append({
                'Model': name,
                'Epochs': len(df),
                'Final mAP@0.5': final_epoch.get('metrics/mAP50(B)', 0),
                'Best mAP@0.5': best_map,
                'Final Precision': final_epoch.get('metrics/precision(B)', 0),
                'Best Precision': best_precision,
                'Final Recall': final_epoch.get('metrics/recall(B)', 0),
                'Best Recall': best_recall,
                'Training Time (min)': final_epoch.get('time', 0) / 60
            })
    
    # Convert to DataFrame for display
    summary_df = pd.DataFrame(summary_data)
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.round(4).values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15] * len(summary_df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the rows
    colors = ['#FFE5E5', '#E5F7F5', '#E5F2FF']
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor(colors[i-1])
    
    plt.tight_layout()
    plt.savefig('epoch_curves_summary.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: epoch_curves_summary.png")
    return fig

def create_improvement_analysis(data):
    """Analyze improvements between training runs"""
    if len(data) < 2:
        print("ℹ️ Need at least 2 training runs for comparison")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔄 Training Improvement Analysis', fontsize=16, fontweight='bold')
    
    # Get original and fixed training data
    original_data = None
    fixed_data = None
    
    for name, df in data.items():
        if 'Original' in name:
            original_data = df
        elif 'Fixed' in name:
            fixed_data = df
    
    if original_data is None or fixed_data is None:
        print("ℹ️ Could not find both original and fixed training data")
        return None
    
    # Plot key metrics comparison
    metrics_to_compare = [
        ('metrics/precision(B)', 'Precision Improvement', '🎯'),
        ('metrics/recall(B)', 'Recall Comparison', '🔍'),
        ('metrics/mAP50(B)', 'mAP@0.5 Improvement', '🏆'),
        ('metrics/mAP50-95(B)', 'mAP@0.5:0.95 Improvement', '⭐')
    ]
    
    for idx, (metric, title, emoji) in enumerate(metrics_to_compare):
        ax = axes[idx // 2, idx % 2]
        
        # Plot original training (limited to same length as fixed for fair comparison)
        max_epochs = min(len(original_data), len(fixed_data))
        orig_epochs = range(1, len(original_data) + 1)
        fixed_epochs = range(1, len(fixed_data) + 1)
        
        ax.plot(orig_epochs, original_data[metric], 
               label='Original Training', 
               color='#FF6B6B', linewidth=3, alpha=0.7)
        
        ax.plot(fixed_epochs, fixed_data[metric], 
               label='Fixed Training', 
               color='#4ECDC4', linewidth=3, alpha=0.8)
        
        # Highlight improvement area
        if len(fixed_data) >= len(original_data):
            improvement_start = len(original_data)
            remaining_epochs = range(improvement_start + 1, len(fixed_data) + 1)
            if len(remaining_epochs) > 0:
                ax.fill_between(remaining_epochs, 
                               0, fixed_data[metric].iloc[improvement_start:],
                               alpha=0.2, color='#4ECDC4', 
                               label='Additional Training')
        
        ax.set_title(f'{emoji} {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.split('/')[-1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add improvement percentage
        orig_final = original_data[metric].iloc[-1]
        fixed_final = fixed_data[metric].iloc[-1]
        if orig_final > 0:
            improvement = ((fixed_final - orig_final) / orig_final) * 100
            ax.text(0.02, 0.98, f'Improvement: {improvement:+.1f}%', 
                   transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                   verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('epoch_curves_improvement.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: epoch_curves_improvement.png")
    return fig

def main():
    """Main function to create all epoch curves"""
    print("🚀 Creating Comprehensive Epoch Curves")
    print("=" * 50)
    
    # Load all training data
    data = load_training_data()
    
    if not data:
        print("❌ No training data found!")
        return
    
    print(f"\n📊 Creating visualizations for {len(data)} training runs...")
    
    # Create all visualizations
    figs = []
    
    # 1. Loss curves
    print("\n1️⃣ Creating loss curves...")
    fig1 = create_loss_curves(data)
    figs.append(fig1)
    
    # 2. Performance metrics
    print("\n2️⃣ Creating performance metrics...")
    fig2 = create_performance_metrics(data)
    figs.append(fig2)
    
    # 3. Learning rate curves
    print("\n3️⃣ Creating learning rate curves...")
    fig3 = create_learning_rate_curves(data)
    figs.append(fig3)
    
    # 4. Segmentation comparison (if available)
    print("\n4️⃣ Creating segmentation analysis...")
    fig4 = create_segmentation_comparison(data)
    if fig4:
        figs.append(fig4)
    
    # 5. Training summary
    print("\n5️⃣ Creating training summary...")
    fig5 = create_training_summary(data)
    figs.append(fig5)
    
    # 6. Improvement analysis
    print("\n6️⃣ Creating improvement analysis...")
    fig6 = create_improvement_analysis(data)
    if fig6:
        figs.append(fig6)
    
    print(f"\n✅ Successfully created {len([f for f in figs if f is not None])} visualizations!")
    print("\n📁 Generated files:")
    print("   • epoch_curves_losses.png - Training & validation loss curves")
    print("   • epoch_curves_metrics.png - Performance metrics over time")
    print("   • epoch_curves_learning_rate.png - Learning rate schedules")
    print("   • epoch_curves_summary.png - Comprehensive summary table")
    
    if any('Segmentation' in name for name in data.keys()):
        print("   • epoch_curves_segmentation.png - Segmentation vs detection comparison")
    
    if len(data) >= 2:
        print("   • epoch_curves_improvement.png - Training improvement analysis")
    
    print("\n🎉 All epoch curves generated successfully!")
    
    # Display plots
    plt.show()

if __name__ == "__main__":
    main()
