#!/usr/bin/env python3
"""
YOLOv8 Training Improvements Analysis
For crack_augmented and spalling_rebar datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def calculate_f1_score(precision, recall):
    """Calculate F1 score"""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def analyze_single_training(results_path, dataset_name):
    """Analyze a single training run"""
    
    if not Path(results_path).exists():
        print(f"❌ Results not found: {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    
    print(f"\n{'='*80}")
    print(f"📊 ANALYSIS: {dataset_name}")
    print(f"{'='*80}")
    print(f"Training Path: {results_path}")
    print(f"Total Epochs: {len(df)}\n")
    
    # Determine if detection or segmentation
    is_segmentation = 'metrics/precision(M)' in df.columns
    suffix = '(M)' if is_segmentation else '(B)'
    model_type = "Segmentation" if is_segmentation else "Detection"
    print(f"Model Type: YOLOv8 {model_type}\n")
    
    # Calculate F1 scores
    precision_col = f'metrics/precision{suffix}'
    recall_col = f'metrics/recall{suffix}'
    mAP50_col = f'metrics/mAP50{suffix}'
    mAP50_95_col = f'metrics/mAP50-95{suffix}'
    
    df['f1_score'] = df.apply(
        lambda row: calculate_f1_score(row[precision_col], row[recall_col]), 
        axis=1
    )
    
    # ============== LOSS ANALYSIS ==============
    print("📉 LOSS IMPROVEMENTS:")
    print("-" * 80)
    
    # Initial vs Final losses
    initial_box_loss = df['train/box_loss'].iloc[0]
    final_box_loss = df['train/box_loss'].iloc[-1]
    best_box_loss = df['train/box_loss'].min()
    
    initial_cls_loss = df['train/cls_loss'].iloc[0]
    final_cls_loss = df['train/cls_loss'].iloc[-1]
    best_cls_loss = df['train/cls_loss'].min()
    
    initial_dfl_loss = df['train/dfl_loss'].iloc[0]
    final_dfl_loss = df['train/dfl_loss'].iloc[-1]
    best_dfl_loss = df['train/dfl_loss'].min()
    
    # Segmentation loss if available
    if 'train/seg_loss' in df.columns:
        initial_seg_loss = df['train/seg_loss'].iloc[0]
        final_seg_loss = df['train/seg_loss'].iloc[-1]
        best_seg_loss = df['train/seg_loss'].min()
    
    print(f"📊 Box Loss:")
    print(f"   Initial:     {initial_box_loss:.6f}")
    print(f"   Final:       {final_box_loss:.6f}")
    print(f"   Best:        {best_box_loss:.6f}")
    print(f"   Reduction:   {((initial_box_loss - final_box_loss) / initial_box_loss * 100):.2f}%\n")
    
    print(f"📊 Classification Loss:")
    print(f"   Initial:     {initial_cls_loss:.6f}")
    print(f"   Final:       {final_cls_loss:.6f}")
    print(f"   Best:        {best_cls_loss:.6f}")
    print(f"   Reduction:   {((initial_cls_loss - final_cls_loss) / initial_cls_loss * 100):.2f}%\n")
    
    print(f"📊 DFL Loss:")
    print(f"   Initial:     {initial_dfl_loss:.6f}")
    print(f"   Final:       {final_dfl_loss:.6f}")
    print(f"   Best:        {best_dfl_loss:.6f}")
    print(f"   Reduction:   {((initial_dfl_loss - final_dfl_loss) / initial_dfl_loss * 100):.2f}%\n")
    
    if 'train/seg_loss' in df.columns:
        print(f"📊 Segmentation Loss:")
        print(f"   Initial:     {initial_seg_loss:.6f}")
        print(f"   Final:       {final_seg_loss:.6f}")
        print(f"   Best:        {best_seg_loss:.6f}")
        print(f"   Reduction:   {((initial_seg_loss - final_seg_loss) / initial_seg_loss * 100):.2f}%\n")
    
    # Total loss
    if 'train/seg_loss' in df.columns:
        total_initial = initial_box_loss + initial_cls_loss + initial_dfl_loss + initial_seg_loss
        total_final = final_box_loss + final_cls_loss + final_dfl_loss + final_seg_loss
    else:
        total_initial = initial_box_loss + initial_cls_loss + initial_dfl_loss
        total_final = final_box_loss + final_cls_loss + final_dfl_loss
    
    print(f"📊 TOTAL LOSS:")
    print(f"   Initial:     {total_initial:.6f}")
    print(f"   Final:       {total_final:.6f}")
    print(f"   Reduction:   {((total_initial - total_final) / total_initial * 100):.2f}%\n")
    
    # ============== PERFORMANCE METRICS ==============
    print("\n🎯 PERFORMANCE METRICS IMPROVEMENTS:")
    print("-" * 80)
    
    metrics = {
        'Precision': precision_col,
        'Recall': recall_col,
        'mAP@0.5': mAP50_col,
        'mAP@0.5:0.95': mAP50_95_col,
        'F1 Score': 'f1_score'
    }
    
    summary_data = []
    
    for metric_name, col in metrics.items():
        initial = df[col].iloc[0]
        final = df[col].iloc[-1]
        best = df[col].max()
        best_epoch = df[col].idxmax() + 1
        improvement = ((final - initial) / initial * 100) if initial > 0 else float('inf')
        
        summary_data.append({
            'Metric': metric_name,
            'Initial': initial,
            'Final': final,
            'Best': best,
            'Best Epoch': best_epoch,
            'Improvement': improvement
        })
        
        print(f"📊 {metric_name}:")
        print(f"   Initial:     {initial:.4f}")
        print(f"   Final:       {final:.4f}")
        print(f"   Best:        {best:.4f} (Epoch {best_epoch})")
        print(f"   Improvement: {improvement:+.2f}%\n")
    
    # ============== F1-CONFIDENCE ANALYSIS ==============
    print("\n🎯 F1-CONFIDENCE CORRELATION:")
    print("-" * 80)
    
    correlation = df['f1_score'].corr(df[mAP50_col])
    print(f"Correlation: {correlation:.4f}")
    
    if correlation > 0.7:
        print("✅ Strong positive correlation - Model is well-balanced")
    elif correlation > 0.5:
        print("🟡 Moderate correlation - Good learning pattern")
    else:
        print("⚠️ Weak correlation - Check if model is overfitting")
    
    # Best performance point
    best_f1_epoch = df['f1_score'].idxmax() + 1
    best_map_epoch = df[mAP50_col].idxmax() + 1
    
    print(f"\n📊 Best F1 Score Point (Epoch {best_f1_epoch}):")
    print(f"   F1 Score:    {df['f1_score'].iloc[best_f1_epoch-1]:.4f}")
    print(f"   Precision:   {df[precision_col].iloc[best_f1_epoch-1]:.4f}")
    print(f"   Recall:      {df[recall_col].iloc[best_f1_epoch-1]:.4f}")
    print(f"   mAP@0.5:     {df[mAP50_col].iloc[best_f1_epoch-1]:.4f}")
    
    print(f"\n📊 Best Confidence Point (Epoch {best_map_epoch}):")
    print(f"   mAP@0.5:     {df[mAP50_col].iloc[best_map_epoch-1]:.4f}")
    print(f"   F1 Score:    {df['f1_score'].iloc[best_map_epoch-1]:.4f}")
    print(f"   Precision:   {df[precision_col].iloc[best_map_epoch-1]:.4f}")
    print(f"   Recall:      {df[recall_col].iloc[best_map_epoch-1]:.4f}")
    
    # ============== EPOCH BY EPOCH PROGRESSION ==============
    print(f"\n📈 EPOCH-BY-EPOCH PROGRESSION (Every 10 epochs):")
    print("-" * 80)
    print(f"{'Epoch':<8} {'Box Loss':<12} {'Cls Loss':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'mAP@0.5':<10}")
    print("-" * 80)
    
    step = max(1, len(df)//10)
    for idx in range(0, len(df), step):
        epoch = idx + 1
        row = df.iloc[idx]
        print(f"{epoch:<8} {row['train/box_loss']:<12.6f} {row['train/cls_loss']:<12.6f} "
              f"{row[precision_col]:<12.4f} {row[recall_col]:<12.4f} "
              f"{row['f1_score']:<10.4f} {row[mAP50_col]:<10.4f}")
    
    # Show last epoch if not shown
    if step > 0 and len(df) % step != 1:
        row = df.iloc[-1]
        print(f"{len(df):<8} {row['train/box_loss']:<12.6f} {row['train/cls_loss']:<12.6f} "
              f"{row[precision_col]:<12.4f} {row[recall_col]:<12.4f} "
              f"{row['f1_score']:<10.4f} {row[mAP50_col]:<10.4f}")
    
    # ============== FINAL ASSESSMENT ==============
    print(f"\n🏆 FINAL ASSESSMENT:")
    print("-" * 80)
    
    final_f1 = df['f1_score'].iloc[-1]
    final_map = df[mAP50_col].iloc[-1]
    
    if final_f1 >= 0.7 and final_map >= 0.6:
        rating = "🏆 EXCELLENT - Production Ready"
    elif final_f1 >= 0.5 and final_map >= 0.4:
        rating = "✅ GOOD - Acceptable Performance"
    elif final_f1 >= 0.3 and final_map >= 0.2:
        rating = "🟡 MODERATE - Needs Fine-tuning"
    else:
        rating = "🔶 NEEDS IMPROVEMENT - More Training Required"
    
    print(f"Overall Rating: {rating}")
    print(f"Final F1 Score: {final_f1:.4f}")
    print(f"Final mAP@0.5:  {final_map:.4f}")
    
    return {
        'df': df,
        'dataset_name': dataset_name,
        'suffix': suffix,
        'summary': summary_data
    }

def create_comparison_visualization(results_dict):
    """Create comparison visualization for all datasets"""
    
    if not results_dict:
        print("No results to visualize")
        return
    
    num_datasets = len(results_dict)
    fig = plt.figure(figsize=(20, 6 * num_datasets))
    
    for idx, (dataset_name, data) in enumerate(results_dict.items()):
        df = data['df']
        suffix = data['suffix']
        epochs = range(1, len(df) + 1)
        
        # Create subplot grid for this dataset
        base_idx = idx * 6
        
        # 1. Losses
        ax1 = plt.subplot(num_datasets, 6, base_idx + 1)
        ax1.plot(epochs, df['train/box_loss'], label='Box Loss', linewidth=2)
        ax1.plot(epochs, df['train/cls_loss'], label='Class Loss', linewidth=2)
        ax1.plot(epochs, df['train/dfl_loss'], label='DFL Loss', linewidth=2)
        if 'train/seg_loss' in df.columns:
            ax1.plot(epochs, df['train/seg_loss'], label='Seg Loss', linewidth=2)
        ax1.set_title(f'{dataset_name}\n📉 Training Losses', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision & Recall
        ax2 = plt.subplot(num_datasets, 6, base_idx + 2)
        ax2.plot(epochs, df[f'metrics/precision{suffix}'], label='Precision', 
                linewidth=3, marker='o', markersize=3)
        ax2.plot(epochs, df[f'metrics/recall{suffix}'], label='Recall', 
                linewidth=3, marker='s', markersize=3)
        ax2.set_title('🎯 Precision & Recall', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. F1 Score
        ax3 = plt.subplot(num_datasets, 6, base_idx + 3)
        ax3.plot(epochs, df['f1_score'], label='F1 Score', 
                linewidth=3, marker='o', markersize=3, color='green')
        best_f1_idx = df['f1_score'].idxmax()
        ax3.scatter(best_f1_idx + 1, df['f1_score'].iloc[best_f1_idx],
                   s=200, color='red', zorder=5, label=f'Best: {df["f1_score"].iloc[best_f1_idx]:.4f}')
        ax3.set_title('🎯 F1 Score', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. mAP Scores
        ax4 = plt.subplot(num_datasets, 6, base_idx + 4)
        ax4.plot(epochs, df[f'metrics/mAP50{suffix}'], label='mAP@0.5',
                linewidth=3, marker='o', markersize=3)
        ax4.plot(epochs, df[f'metrics/mAP50-95{suffix}'], label='mAP@0.5:0.95',
                linewidth=3, marker='s', markersize=3)
        ax4.set_title('🏆 mAP Scores', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('mAP')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # 5. F1 vs Confidence
        ax5 = plt.subplot(num_datasets, 6, base_idx + 5)
        scatter = ax5.scatter(df[f'metrics/mAP50{suffix}'], df['f1_score'],
                            c=epochs, cmap='viridis', s=80, alpha=0.6)
        plt.colorbar(scatter, ax=ax5, label='Epoch')
        # Trend line
        z = np.polyfit(df[f'metrics/mAP50{suffix}'], df['f1_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[f'metrics/mAP50{suffix}'].min(), 
                            df[f'metrics/mAP50{suffix}'].max(), 100)
        ax5.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax5.set_title('🎯 F1 vs Confidence', fontweight='bold')
        ax5.set_xlabel('Confidence (mAP@0.5)')
        ax5.set_ylabel('F1 Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Total Loss
        ax6 = plt.subplot(num_datasets, 6, base_idx + 6)
        if 'train/seg_loss' in df.columns:
            total_train = (df['train/box_loss'] + df['train/cls_loss'] + 
                          df['train/dfl_loss'] + df['train/seg_loss'])
            total_val = (df['val/box_loss'] + df['val/cls_loss'] + 
                        df['val/dfl_loss'] + df['val/seg_loss'])
        else:
            total_train = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
            total_val = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
        
        ax6.plot(epochs, total_train, label='Train Total', linewidth=3)
        ax6.plot(epochs, total_val, label='Val Total', linewidth=3)
        ax6.set_title('📉 Total Loss', fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Total Loss')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('🔍 YOLOv8 Training Analysis - All Datasets', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = 'yolov8_training_improvements_complete.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Complete visualization saved: {output_path}")

def main():
    """Main analysis function"""
    
    print("🚀 YOLOv8 TRAINING IMPROVEMENTS ANALYSIS")
    print("="*80)
    print("Analyzing crack_augmented and spalling_rebar datasets\n")
    
    # Define training results to analyze
    training_results = {
        'Crack Augmented (Run 1)': 'runs/segment/yolov8_crack_augmented/results.csv',
        'Crack Augmented (Run 2)': 'runs/segment/yolov8_crack_augmented2/results.csv',
        'Spalling & Rebar (Final)': 'runs/segment/yolov8_spalling_rebar_final/results.csv',
        'Spalling & Rebar (Final 2)': 'runs/segment/yolov8_spalling_rebar_final2/results.csv',
        'Spalling & Rebar (Final 3)': 'runs/segment/yolov8_spalling_rebar_final3/results.csv',
        'Spalling & Rebar (Segmentation)': 'runs/segment/yolov8_spalling_rebar_segmentation/results.csv'
    }
    
    results_dict = {}
    
    for dataset_name, path in training_results.items():
        result = analyze_single_training(path, dataset_name)
        if result:
            results_dict[dataset_name] = result
    
    # Create visualizations
    if results_dict:
        print(f"\n{'='*80}")
        print("📊 CREATING COMPARISON VISUALIZATIONS")
        print('='*80)
        create_comparison_visualization(results_dict)
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print('='*80)
    print("\nGenerated files:")
    print("  📊 yolov8_training_improvements_complete.png - Complete visualization")

if __name__ == "__main__":
    main()

