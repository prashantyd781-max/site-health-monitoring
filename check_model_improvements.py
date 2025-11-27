#!/usr/bin/env python3
"""
Quick Model Improvement Checker
Analyzes losses and F1 confidence scores from training results
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def calculate_f1_score(precision, recall):
    """Calculate F1 score from precision and recall"""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def analyze_training_improvements(results_path):
    """Analyze training improvements from results.csv"""
    
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    print("🔍 ANALYZING MODEL TRAINING IMPROVEMENTS")
    print("=" * 80)
    
    # Load training results
    df = pd.read_csv(results_path)
    print(f"✅ Loaded {len(df)} epochs of training data\n")
    
    # ============== LOSS ANALYSIS ==============
    print("📉 LOSS IMPROVEMENTS:")
    print("-" * 80)
    
    loss_columns = {
        'train/box_loss': 'Box Loss (Train)',
        'val/box_loss': 'Box Loss (Val)',
        'train/cls_loss': 'Classification Loss (Train)',
        'val/cls_loss': 'Classification Loss (Val)',
        'train/dfl_loss': 'DFL Loss (Train)',
        'val/dfl_loss': 'DFL Loss (Val)'
    }
    
    for col, name in loss_columns.items():
        if col in df.columns:
            initial = df[col].iloc[0]
            final = df[col].iloc[-1]
            best = df[col].min()
            improvement = ((initial - final) / initial) * 100 if initial > 0 else 0
            
            print(f"📊 {name}:")
            print(f"   Initial: {initial:.6f}")
            print(f"   Final:   {final:.6f}")
            print(f"   Best:    {best:.6f}")
            print(f"   Improvement: {improvement:+.2f}%")
            print()
    
    # Total loss
    if all(col in df.columns for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
        total_train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
        total_val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
        
        print("📊 TOTAL COMBINED LOSS:")
        print(f"   Train Loss - Initial: {total_train_loss.iloc[0]:.6f}, Final: {total_train_loss.iloc[-1]:.6f}")
        print(f"   Val Loss   - Initial: {total_val_loss.iloc[0]:.6f}, Final: {total_val_loss.iloc[-1]:.6f}")
        train_improvement = ((total_train_loss.iloc[0] - total_train_loss.iloc[-1]) / total_train_loss.iloc[0]) * 100
        val_improvement = ((total_val_loss.iloc[0] - total_val_loss.iloc[-1]) / total_val_loss.iloc[0]) * 100
        print(f"   Train Improvement: {train_improvement:+.2f}%")
        print(f"   Val Improvement: {val_improvement:+.2f}%")
        print()
    
    # ============== METRICS ANALYSIS ==============
    print("\n🎯 PERFORMANCE METRICS IMPROVEMENTS:")
    print("-" * 80)
    
    # Determine if detection or segmentation model
    is_segmentation = 'metrics/precision(M)' in df.columns
    metric_suffix = '(M)' if is_segmentation else '(B)'
    model_type = "Segmentation" if is_segmentation else "Detection"
    
    print(f"Model Type: {model_type}\n")
    
    metrics = {
        f'metrics/precision{metric_suffix}': 'Precision',
        f'metrics/recall{metric_suffix}': 'Recall',
        f'metrics/mAP50{metric_suffix}': 'mAP@0.5',
        f'metrics/mAP50-95{metric_suffix}': 'mAP@0.5:0.95'
    }
    
    best_metrics = {}
    
    for col, name in metrics.items():
        if col in df.columns:
            initial = df[col].iloc[0]
            final = df[col].iloc[-1]
            best = df[col].max()
            best_epoch = df[col].idxmax() + 1
            improvement = ((final - initial) / initial) * 100 if initial > 0 else float('inf')
            
            best_metrics[name] = best
            
            print(f"📊 {name}:")
            print(f"   Initial: {initial:.4f}")
            print(f"   Final:   {final:.4f}")
            print(f"   Best:    {best:.4f} (Epoch {best_epoch})")
            print(f"   Improvement: {improvement:+.2f}%")
            print()
    
    # ============== F1 SCORE ANALYSIS ==============
    print("\n🎯 F1 SCORE IMPROVEMENTS:")
    print("-" * 80)
    
    precision_col = f'metrics/precision{metric_suffix}'
    recall_col = f'metrics/recall{metric_suffix}'
    
    if precision_col in df.columns and recall_col in df.columns:
        # Calculate F1 scores for all epochs
        df['f1_score'] = df.apply(
            lambda row: calculate_f1_score(row[precision_col], row[recall_col]), 
            axis=1
        )
        
        initial_f1 = df['f1_score'].iloc[0]
        final_f1 = df['f1_score'].iloc[-1]
        best_f1 = df['f1_score'].max()
        best_f1_epoch = df['f1_score'].idxmax() + 1
        f1_improvement = ((final_f1 - initial_f1) / initial_f1) * 100 if initial_f1 > 0 else float('inf')
        
        print(f"📊 F1 Score:")
        print(f"   Initial: {initial_f1:.4f}")
        print(f"   Final:   {final_f1:.4f}")
        print(f"   Best:    {best_f1:.4f} (Epoch {best_f1_epoch})")
        print(f"   Improvement: {f1_improvement:+.2f}%")
        print()
        
        # Show F1 score per epoch
        print("📈 F1 Score Progression:")
        for idx, row in df.iterrows():
            epoch = idx + 1
            f1 = row['f1_score']
            marker = "⭐" if f1 == best_f1 else "  "
            print(f"   {marker} Epoch {epoch:3d}: F1 = {f1:.4f}")
    
    # ============== CONFIDENCE ANALYSIS ==============
    print("\n🎯 CONFIDENCE (mAP) VS F1 ANALYSIS:")
    print("-" * 80)
    
    mAP_col = f'metrics/mAP50{metric_suffix}'
    if mAP_col in df.columns and 'f1_score' in df.columns:
        # Find correlation
        correlation = df['f1_score'].corr(df[mAP_col])
        print(f"📊 F1-Confidence Correlation: {correlation:.4f}")
        print(f"   {'✅ Strong positive correlation' if correlation > 0.7 else '⚠️ Moderate correlation' if correlation > 0.5 else '❌ Weak correlation'}")
        print()
        
        # Best confidence epoch
        best_map = df[mAP_col].max()
        best_map_epoch = df[mAP_col].idxmax() + 1
        f1_at_best_map = df['f1_score'].iloc[best_map_epoch - 1]
        
        print(f"📊 At Best Confidence (Epoch {best_map_epoch}):")
        print(f"   mAP@0.5: {best_map:.4f}")
        print(f"   F1 Score: {f1_at_best_map:.4f}")
        print(f"   Precision: {df[precision_col].iloc[best_map_epoch - 1]:.4f}")
        print(f"   Recall: {df[recall_col].iloc[best_map_epoch - 1]:.4f}")
        print()
    
    # ============== VISUALIZATION ==============
    create_improvement_plots(df, metric_suffix, model_type)
    
    # ============== SUMMARY ==============
    print("\n" + "=" * 80)
    print("📋 SUMMARY:")
    print("=" * 80)
    
    # Overall assessment
    if 'f1_score' in df.columns:
        final_f1 = df['f1_score'].iloc[-1]
        if final_f1 >= 0.7:
            rating = "🏆 EXCELLENT"
        elif final_f1 >= 0.5:
            rating = "✅ GOOD"
        elif final_f1 >= 0.3:
            rating = "🟡 MODERATE"
        else:
            rating = "🔶 NEEDS IMPROVEMENT"
        
        print(f"Model Performance: {rating}")
        print(f"Final F1 Score: {final_f1:.4f}")
    
    if best_metrics:
        print(f"\nBest Metrics Achieved:")
        for name, value in best_metrics.items():
            print(f"   {name}: {value:.4f}")

def create_improvement_plots(df, metric_suffix, model_type):
    """Create visualization plots"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Loss curves
    ax1 = plt.subplot(3, 2, 1)
    epochs = range(1, len(df) + 1)
    
    if 'train/box_loss' in df.columns:
        ax1.plot(epochs, df['train/box_loss'], label='Box Loss (Train)', linewidth=2)
        ax1.plot(epochs, df['val/box_loss'], label='Box Loss (Val)', linewidth=2, linestyle='--')
        ax1.plot(epochs, df['train/cls_loss'], label='Class Loss (Train)', linewidth=2)
        ax1.plot(epochs, df['val/cls_loss'], label='Class Loss (Val)', linewidth=2, linestyle='--')
        ax1.plot(epochs, df['train/dfl_loss'], label='DFL Loss (Train)', linewidth=2)
        ax1.plot(epochs, df['val/dfl_loss'], label='DFL Loss (Val)', linewidth=2, linestyle='--')
    
    ax1.set_title('📉 Training Losses Over Time', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision and Recall
    ax2 = plt.subplot(3, 2, 2)
    precision_col = f'metrics/precision{metric_suffix}'
    recall_col = f'metrics/recall{metric_suffix}'
    
    if precision_col in df.columns:
        ax2.plot(epochs, df[precision_col], label='Precision', linewidth=3, marker='o')
        ax2.plot(epochs, df[recall_col], label='Recall', linewidth=3, marker='s')
    
    ax2.set_title('🎯 Precision & Recall', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. mAP scores
    ax3 = plt.subplot(3, 2, 3)
    mAP50_col = f'metrics/mAP50{metric_suffix}'
    mAP50_95_col = f'metrics/mAP50-95{metric_suffix}'
    
    if mAP50_col in df.columns:
        ax3.plot(epochs, df[mAP50_col], label='mAP@0.5', linewidth=3, marker='o')
        ax3.plot(epochs, df[mAP50_95_col], label='mAP@0.5:0.95', linewidth=3, marker='s')
    
    ax3.set_title('🏆 mAP Scores', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mAP')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. F1 Score
    ax4 = plt.subplot(3, 2, 4)
    if 'f1_score' in df.columns:
        ax4.plot(epochs, df['f1_score'], label='F1 Score', 
                linewidth=3, marker='o', color='green')
        # Mark best F1
        best_f1_idx = df['f1_score'].idxmax()
        ax4.scatter(best_f1_idx + 1, df['f1_score'].iloc[best_f1_idx], 
                   s=200, color='red', zorder=5, label=f'Best F1: {df["f1_score"].iloc[best_f1_idx]:.4f}')
    
    ax4.set_title('🎯 F1 Score Over Time', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # 5. F1 vs Confidence (mAP)
    ax5 = plt.subplot(3, 2, 5)
    if 'f1_score' in df.columns and mAP50_col in df.columns:
        scatter = ax5.scatter(df[mAP50_col], df['f1_score'], 
                             c=epochs, cmap='viridis', s=100, alpha=0.6)
        plt.colorbar(scatter, ax=ax5, label='Epoch')
        
        # Add trend line
        z = np.polyfit(df[mAP50_col], df['f1_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[mAP50_col].min(), df[mAP50_col].max(), 100)
        ax5.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    ax5.set_title('🎯 F1 Score vs Confidence (mAP@0.5)', fontweight='bold')
    ax5.set_xlabel('Confidence (mAP@0.5)')
    ax5.set_ylabel('F1 Score')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Total Loss (Train vs Val)
    ax6 = plt.subplot(3, 2, 6)
    if all(col in df.columns for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
        total_train = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
        total_val = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
        
        ax6.plot(epochs, total_train, label='Total Train Loss', linewidth=3)
        ax6.plot(epochs, total_val, label='Total Val Loss', linewidth=3)
    
    ax6.set_title('📉 Total Combined Loss', fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Total Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'🔍 {model_type} Model Training Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    output_path = 'model_improvement_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {output_path}")
    
    plt.show()

def main():
    """Main function"""
    
    # Check multiple possible locations for results
    possible_paths = [
        "runs/detect/train/results.csv",
        "runs/detect/crack_detection_train/results.csv",
        "runs/detect/crack_detection_fixed/results.csv",
        "runs/segment/train/results.csv",
        "segmentation_model/results.csv",
    ]
    
    print("🔍 Searching for training results...")
    print()
    
    found_results = []
    for path in possible_paths:
        if Path(path).exists():
            found_results.append(path)
            print(f"✅ Found: {path}")
    
    if not found_results:
        print("❌ No training results found!")
        print("\nPlease specify the path to your results.csv file:")
        print("Usage: python check_model_improvements.py")
        print("\nOr modify the 'possible_paths' list in the script.")
        return
    
    print(f"\n📊 Analyzing {len(found_results)} training result(s)...\n")
    
    for idx, path in enumerate(found_results, 1):
        print(f"\n{'='*80}")
        print(f"ANALYSIS {idx}/{len(found_results)}: {path}")
        print('='*80)
        analyze_training_improvements(path)
        if idx < len(found_results):
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()


