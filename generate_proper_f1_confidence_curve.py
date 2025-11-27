import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_crack_f1_curve():
    """
    Generate F1-confidence curve specifically for crack class with light green color
    """
    
    # Path to results file
    results_path = "segmentation_model/results.csv"
    output_dir = "segmentation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found at {results_path}")
        return
    
    # Read the training results
    df = pd.read_csv(results_path)
    
    # Extract metrics for crack class
    epochs = df['epoch']
    precision = df['metrics/precision(M)']
    recall = df['metrics/recall(M)']
    
    # Calculate F1 Score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = f1_scores.fillna(0)
    
    # Confidence proxy
    confidence = df['metrics/mAP50(M)']
    
    # Plot F1 vs Confidence
    plt.figure(figsize=(10, 7))
    plt.plot(confidence, f1_scores, 
             color='lightgreen', linewidth=3, marker='o',
             markerfacecolor='green', markeredgecolor='darkgreen',
             label='Crack F1-Score')
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.title('F1-Confidence Curve for Crack Detection', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence (mAP@0.5)', fontsize=13)
    plt.ylabel('F1 Score', fontsize=13)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='lower right', fontsize=12)
    
    # Highlight best F1
    best_idx = f1_scores.idxmax()
    plt.annotate(f'Best F1: {f1_scores.iloc[best_idx]:.3f}\nEpoch: {epochs.iloc[best_idx]}',
                 xy=(confidence.iloc[best_idx], f1_scores.iloc[best_idx]),
                 xytext=(confidence.iloc[best_idx] + 0.1, f1_scores.iloc[best_idx] - 0.1),
                 arrowprops=dict(arrowstyle='->', color='darkgreen'))
    
    # Save figure
    output_path = os.path.join(output_dir, 'crack_f1_confidence_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Crack F1-Confidence curve saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    generate_crack_f1_curve()
