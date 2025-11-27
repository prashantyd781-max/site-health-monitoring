import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def generate_crack_f1_curve():
    """
    Generate F1-confidence curve specifically for crack class with light green color
    """
    
    # Path to results file
    results_path = "segmentation_model/results.csv"
    output_dir = "segmentation_outputs"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found at {results_path}")
        return
    
    # Read the training results
    df = pd.read_csv(results_path)
    
    # Extract relevant metrics for crack class (Mask metrics)
    epochs = df['epoch']
    precision_mask = df['metrics/precision(M)']
    recall_mask = df['metrics/recall(M)']
    
    # Calculate F1 score from precision and recall
    f1_scores = 2 * (precision_mask * recall_mask) / (precision_mask + recall_mask)
    # Handle division by zero
    f1_scores = f1_scores.fillna(0)
    
    # Create confidence values (using mAP50 as confidence proxy)
    confidence = df['metrics/mAP50(M)']
    
    # Create the plot with custom styling
    plt.figure(figsize=(12, 8))
    
    # Plot F1-Confidence curve with light green color
    plt.plot(confidence, f1_scores, 
             color='lightgreen', 
             linewidth=3, 
             marker='o', 
             markersize=4, 
             markerfacecolor='green',
             markeredgecolor='darkgreen',
             markeredgewidth=1,
             label='Crack F1-Score')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Customize the plot
    plt.title('F1-Confidence Curve for Crack Detection\n(Segmentation Model)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Confidence (mAP@0.5)', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend(fontsize=12, loc='lower right')
    
    # Add annotations for best F1 score
    best_f1_idx = f1_scores.idxmax()
    best_f1 = f1_scores.iloc[best_f1_idx]
    best_conf = confidence.iloc[best_f1_idx]
    best_epoch = epochs.iloc[best_f1_idx]
    
    plt.annotate(f'Best F1: {best_f1:.3f}\nEpoch: {best_epoch}\nConfidence: {best_conf:.3f}',
                xy=(best_conf, best_f1), 
                xytext=(best_conf + 0.1, best_f1 - 0.1),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Add some statistics text
    final_f1 = f1_scores.iloc[-1]
    final_conf = confidence.iloc[-1]
    max_f1 = f1_scores.max()
    
    stats_text = f"""Training Statistics:
    • Final F1 Score: {final_f1:.3f}
    • Final Confidence: {final_conf:.3f}
    • Maximum F1 Score: {max_f1:.3f}
    • Total Epochs: {len(epochs)}"""
    
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, 
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'crack_f1_confidence_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Crack F1-Confidence curve saved to: {output_path}")
    
    # Also save a version with higher contrast
    plt.savefig(output_path.replace('.png', '_hd.png'), dpi=600, bbox_inches='tight', facecolor='white')
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Crack Detection F1-Score Analysis:")
    print(f"   • Best F1 Score: {max_f1:.4f} (Epoch {epochs.iloc[best_f1_idx]})")
    print(f"   • Final F1 Score: {final_f1:.4f}")
    print(f"   • Improvement: {((final_f1 - f1_scores.iloc[0]) / f1_scores.iloc[0] * 100):.1f}%")
    print(f"   • Average F1 Score: {f1_scores.mean():.4f}")

def generate_detailed_metrics_plot():
    """
    Generate additional detailed metrics plot for crack detection
    """
    results_path = "segmentation_model/results.csv"
    output_dir = "segmentation_outputs"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found at {results_path}")
        return
    
    df = pd.read_csv(results_path)
    
    # Create subplots for different metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = df['epoch']
    
    # Plot 1: Precision and Recall over epochs
    ax1.plot(epochs, df['metrics/precision(M)'], color='lightgreen', linewidth=2, label='Precision', marker='o', markersize=3)
    ax1.plot(epochs, df['metrics/recall(M)'], color='green', linewidth=2, label='Recall', marker='s', markersize=3)
    ax1.set_title('Crack Detection: Precision & Recall', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score over epochs
    f1_scores = 2 * (df['metrics/precision(M)'] * df['metrics/recall(M)']) / (df['metrics/precision(M)'] + df['metrics/recall(M)'])
    ax2.plot(epochs, f1_scores, color='lightgreen', linewidth=3, label='F1 Score', marker='D', markersize=3)
    ax2.set_title('Crack Detection: F1 Score Progress', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: mAP scores
    ax3.plot(epochs, df['metrics/mAP50(M)'], color='lightgreen', linewidth=2, label='mAP@0.5', marker='o', markersize=3)
    ax3.plot(epochs, df['metrics/mAP50-95(M)'], color='green', linewidth=2, label='mAP@0.5:0.95', marker='s', markersize=3)
    ax3.set_title('Crack Detection: mAP Scores', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mAP')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss over epochs
    ax4.plot(epochs, df['val/seg_loss'], color='red', linewidth=2, label='Segmentation Loss', marker='o', markersize=3)
    ax4.plot(epochs, df['val/cls_loss'], color='orange', linewidth=2, label='Classification Loss', marker='s', markersize=3)
    ax4.set_title('Crack Detection: Validation Losses', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the detailed metrics plot
    detailed_output_path = os.path.join(output_dir, 'crack_detailed_metrics.png')
    plt.savefig(detailed_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Detailed metrics plot saved to: {detailed_output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("🔍 Generating F1-Confidence Curve for Crack Detection...")
    print("=" * 60)
    
    # Generate the main F1-confidence curve
    generate_crack_f1_curve()
    
    print("\n" + "=" * 60)
    print("📈 Generating detailed metrics analysis...")
    
    # Generate detailed metrics plot
    generate_detailed_metrics_plot()
    
    print("\n✅ All visualizations generated successfully!")
    print("📁 Check the 'segmentation_outputs' folder for the generated images.")





