#!/usr/bin/env python3
"""
Generate F1-Confidence Curve for Crack Class Only
Uses existing JSON data to recreate the crack curve in green
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_crack_f1_curve():
    """Generate F1-confidence curve showing only crack class in green"""
    
    # Path to the existing F1 confidence data
    json_path = "segmentation_outputs/f1_confidence_data.json"
    
    if not Path(json_path).exists():
        print(f"❌ Data file not found: {json_path}")
        return
    
    # Load the F1-confidence data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    confidence_thresholds = data['confidence_thresholds']
    f1_scores = data['f1_scores']
    best_f1 = data['best_f1']
    best_confidence = data['best_confidence']
    
    print(f"📊 Loaded data: {len(confidence_thresholds)} confidence thresholds")
    print(f"🎯 Best F1 Score: {best_f1:.3f} at confidence: {best_confidence}")
    
    # Create the figure with similar styling to the original
    plt.figure(figsize=(12, 8))
    
    # Plot the crack F1-confidence curve in green
    plt.plot(confidence_thresholds, f1_scores, 
             color='green', 
             linewidth=2.5, 
             label='crack',
             alpha=0.8)
    
    # Customize the plot to match the original style
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Confidence', fontsize=12, fontweight='bold')
    plt.ylabel('F1', fontsize=12, fontweight='bold')
    plt.title('F1-Confidence Curve', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend in the style of the original
    plt.legend(loc='center right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Add annotation for best F1 score
    plt.annotate(f'Best F1: {best_f1:.3f}', 
                xy=(best_confidence, best_f1),
                xytext=(0.3, 0.8),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    # Save the figure
    output_path = "segmentation_outputs/crack_only_f1_confidence_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Crack F1-Confidence curve saved to: {output_path}")
    
    # Display statistics
    print(f"\n📈 Curve Statistics:")
    print(f"   • Maximum F1 Score: {max(f1_scores):.3f}")
    print(f"   • Minimum F1 Score: {min(f1_scores):.3f}")
    print(f"   • F1 at 0.5 confidence: {f1_scores[confidence_thresholds.index(0.5)]:.3f}")
    
    # Show the plot
    plt.show()
    
    return confidence_thresholds, f1_scores

if __name__ == "__main__":
    print("🚀 Generating Crack F1-Confidence Curve...")
    generate_crack_f1_curve()
    print("✨ Done!")





