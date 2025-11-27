#!/usr/bin/env python3
"""
Create simple visualization charts for YOLOv8 improvements
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def create_best_model_charts():
    """Create charts for the best performing model"""
    
    results_path = 'runs/segment/yolov8_spalling_rebar_segmentation/results.csv'
    df = pd.read_csv(results_path)
    
    # Calculate F1
    precision = df['metrics/precision(M)']
    recall = df['metrics/recall(M)']
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = f1.fillna(0)
    df['f1'] = f1
    
    epochs = range(1, len(df) + 1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss Improvements
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, df['train/box_loss'], label='Box Loss', linewidth=2.5, marker='o')
    ax1.plot(epochs, df['train/cls_loss'], label='Classification Loss', linewidth=2.5, marker='s')
    ax1.plot(epochs, df['train/dfl_loss'], label='DFL Loss', linewidth=2.5, marker='^')
    ax1.plot(epochs, df['train/seg_loss'], label='Segmentation Loss', linewidth=2.5, marker='D')
    ax1.set_title('📉 Training Loss Reductions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add improvement text
    total_initial = (df['train/box_loss'].iloc[0] + df['train/cls_loss'].iloc[0] + 
                    df['train/dfl_loss'].iloc[0] + df['train/seg_loss'].iloc[0])
    total_final = (df['train/box_loss'].iloc[-1] + df['train/cls_loss'].iloc[-1] + 
                  df['train/dfl_loss'].iloc[-1] + df['train/seg_loss'].iloc[-1])
    reduction = ((total_initial - total_final) / total_initial * 100)
    ax1.text(0.05, 0.95, f'Total Loss Reduction: {reduction:.1f}%', 
            transform=ax1.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 2. Precision & Recall
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, precision, label='Precision', linewidth=3, marker='o', markersize=8, color='#2E86DE')
    ax2.plot(epochs, recall, label='Recall', linewidth=3, marker='s', markersize=8, color='#EE5A24')
    ax2.set_title('🎯 Precision & Recall Improvements', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add improvement percentages
    prec_improvement = ((precision.iloc[-1] - precision.iloc[0]) / precision.iloc[0] * 100)
    rec_improvement = ((recall.iloc[-1] - recall.iloc[0]) / recall.iloc[0] * 100)
    ax2.text(0.05, 0.95, f'Precision: +{prec_improvement:.1f}%\nRecall: +{rec_improvement:.1f}%', 
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. F1 Score
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, f1, label='F1 Score', linewidth=3.5, marker='o', markersize=10, 
            color='#10AC84', markerfacecolor='#10AC84', markeredgecolor='darkgreen', markeredgewidth=2)
    best_f1_idx = f1.idxmax()
    ax3.scatter(best_f1_idx + 1, f1.iloc[best_f1_idx], s=400, color='red', zorder=5, 
               marker='*', edgecolors='darkred', linewidths=2, label=f'Best: {f1.iloc[best_f1_idx]:.4f}')
    ax3.set_title('🎯 F1 Score Progression', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Add improvement
    f1_improvement = ((f1.iloc[-1] - f1.iloc[0]) / f1.iloc[0] * 100)
    ax3.text(0.05, 0.95, f'F1 Improvement:\n+{f1_improvement:.1f}%', 
            transform=ax3.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9), fontweight='bold')
    
    # 4. mAP Scores
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, df['metrics/mAP50(M)'], label='mAP@0.5', linewidth=3, 
            marker='o', markersize=8, color='#9B59B6')
    ax4.plot(epochs, df['metrics/mAP50-95(M)'], label='mAP@0.5:0.95', linewidth=3, 
            marker='s', markersize=8, color='#3498DB')
    ax4.set_title('🏆 mAP (Confidence) Improvements', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('mAP', fontsize=12)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Add improvement
    map_improvement = ((df['metrics/mAP50(M)'].iloc[-1] - df['metrics/mAP50(M)'].iloc[0]) / 
                      df['metrics/mAP50(M)'].iloc[0] * 100)
    ax4.text(0.05, 0.95, f'mAP@0.5 Improvement:\n+{map_improvement:.1f}%', 
            transform=ax4.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 5. F1 vs Confidence (mAP)
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(df['metrics/mAP50(M)'], f1, c=epochs, cmap='viridis', 
                         s=200, alpha=0.7, edgecolors='black', linewidths=2)
    cbar = plt.colorbar(scatter, ax=ax5, label='Epoch')
    
    # Add trend line
    z = np.polyfit(df['metrics/mAP50(M)'], f1, 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['metrics/mAP50(M)'].min(), df['metrics/mAP50(M)'].max(), 100)
    ax5.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=3, label='Trend')
    
    ax5.set_title('🎯 F1 vs Confidence Correlation', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Confidence (mAP@0.5)', fontsize=12)
    ax5.set_ylabel('F1 Score', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Add correlation
    correlation = f1.corr(df['metrics/mAP50(M)'])
    ax5.text(0.05, 0.95, f'Correlation: {correlation:.4f}\n✅ Strong Positive', 
            transform=ax5.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # 6. Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary data
    summary_data = [
        ['Metric', 'Initial', 'Final', 'Best', 'Improvement'],
        ['Precision', f'{precision.iloc[0]:.3f}', f'{precision.iloc[-1]:.3f}', 
         f'{precision.max():.3f}', f'+{prec_improvement:.1f}%'],
        ['Recall', f'{recall.iloc[0]:.3f}', f'{recall.iloc[-1]:.3f}', 
         f'{recall.max():.3f}', f'+{rec_improvement:.1f}%'],
        ['F1 Score', f'{f1.iloc[0]:.3f}', f'{f1.iloc[-1]:.3f}', 
         f'{f1.max():.3f}', f'+{f1_improvement:.1f}%'],
        ['mAP@0.5', f'{df["metrics/mAP50(M)"].iloc[0]:.3f}', 
         f'{df["metrics/mAP50(M)"].iloc[-1]:.3f}', 
         f'{df["metrics/mAP50(M)"].max():.3f}', f'+{map_improvement:.1f}%'],
        ['Total Loss', f'{total_initial:.3f}', f'{total_final:.3f}', 
         f'{total_final:.3f}', f'-{reduction:.1f}%']
    ]
    
    table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.15, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style data rows
    colors = ['#ECF0F1', '#D5DBDB', '#ECF0F1', '#D5DBDB', '#ECF0F1']
    for i in range(1, 6):
        for j in range(5):
            table[(i, j)].set_facecolor(colors[i-1])
    
    ax6.set_title('📊 Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Overall title
    plt.suptitle('🏆 YOLOv8 Spalling & Exposed Rebar - Training Improvements Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save
    output_path = 'yolov8_spalling_rebar_improvements.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualization saved: {output_path}")
    
    plt.close()

def create_comparison_chart():
    """Create comparison chart for all three models"""
    
    models = {
        'Segmentation': 'runs/segment/yolov8_spalling_rebar_segmentation/results.csv',
        'Final v2': 'runs/segment/yolov8_spalling_rebar_final2/results.csv',
        'Final v3': 'runs/segment/yolov8_spalling_rebar_final3/results.csv'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['#2ECC71', '#E74C3C', '#F39C12']
    
    for idx, (name, path) in enumerate(models.items()):
        df = pd.read_csv(path)
        
        precision = df['metrics/precision(M)']
        recall = df['metrics/recall(M)']
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = f1.fillna(0)
        
        epochs = range(1, len(df) + 1)
        color = colors[idx]
        
        # F1 Score comparison
        axes[0, 0].plot(epochs, f1, label=name, linewidth=2.5, marker='o', color=color)
        
        # mAP comparison
        axes[0, 1].plot(epochs, df['metrics/mAP50(M)'], label=name, linewidth=2.5, 
                       marker='s', color=color)
        
        # Precision comparison
        axes[1, 0].plot(epochs, precision, label=name, linewidth=2.5, marker='^', color=color)
        
        # Recall comparison
        axes[1, 1].plot(epochs, recall, label=name, linewidth=2.5, marker='D', color=color)
    
    # Configure subplots
    axes[0, 0].set_title('🎯 F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('F1 Score', fontsize=12)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('🏆 mAP@0.5 Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('mAP@0.5', fontsize=12)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('🎯 Precision Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('🔍 Recall Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Recall', fontsize=12)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('📊 YOLOv8 Models Comparison - All Training Runs', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = 'yolov8_models_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparison chart saved: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("🎨 Creating visualizations...")
    print()
    
    print("1️⃣ Creating best model improvement charts...")
    create_best_model_charts()
    
    print("\n2️⃣ Creating models comparison chart...")
    create_comparison_chart()
    
    print("\n✅ All visualizations created successfully!")
    print("\nGenerated files:")
    print("  📊 yolov8_spalling_rebar_improvements.png - Detailed analysis of best model")
    print("  📊 yolov8_models_comparison.png - Comparison across all models")


