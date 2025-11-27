#!/usr/bin/env python3
"""
YOLOv8 Training Improvements Report (Text Only)
For crack_augmented and spalling_rebar datasets
"""

import pandas as pd
from pathlib import Path

def calculate_f1_score(precision, recall):
    """Calculate F1 score"""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def analyze_training(results_path, dataset_name):
    """Analyze a single training run"""
    
    if not Path(results_path).exists():
        print(f"❌ Results not found: {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    
    print(f"\n{'='*80}")
    print(f"📊 {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Path: {results_path}")
    print(f"Epochs Trained: {len(df)}\n")
    
    # Determine model type
    is_segmentation = 'metrics/precision(M)' in df.columns
    suffix = '(M)' if is_segmentation else '(B)'
    model_type = "YOLOv8 Segmentation" if is_segmentation else "YOLOv8 Detection"
    print(f"Model: {model_type}\n")
    
    # Calculate F1
    precision_col = f'metrics/precision{suffix}'
    recall_col = f'metrics/recall{suffix}'
    mAP50_col = f'metrics/mAP50{suffix}'
    mAP50_95_col = f'metrics/mAP50-95{suffix}'
    
    df['f1'] = df.apply(
        lambda row: calculate_f1_score(row[precision_col], row[recall_col]), 
        axis=1
    )
    
    # ==== LOSSES ====
    print("📉 LOSS REDUCTIONS:")
    print("-" * 80)
    
    losses = {
        'Box Loss': 'train/box_loss',
        'Classification Loss': 'train/cls_loss',
        'DFL Loss': 'train/dfl_loss'
    }
    
    if 'train/seg_loss' in df.columns:
        losses['Segmentation Loss'] = 'train/seg_loss'
    
    total_initial = 0
    total_final = 0
    
    for name, col in losses.items():
        initial = df[col].iloc[0]
        final = df[col].iloc[-1]
        best = df[col].min()
        reduction = ((initial - final) / initial * 100) if initial > 0 else 0
        
        total_initial += initial
        total_final += final
        
        print(f"{name:20s}: {initial:8.4f} → {final:8.4f} (Best: {best:8.4f}) | Reduction: {reduction:6.2f}%")
    
    total_reduction = ((total_initial - total_final) / total_initial * 100) if total_initial > 0 else 0
    print(f"{'─' * 80}")
    print(f"{'TOTAL LOSS':20s}: {total_initial:8.4f} → {total_final:8.4f} | Reduction: {total_reduction:6.2f}%\n")
    
    # ==== METRICS ====
    print("🎯 PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"{'Metric':<20s} {'Initial':>10s} {'Final':>10s} {'Best':>10s} {'Best Epoch':>12s} {'Change':>10s}")
    print("-" * 80)
    
    metrics = {
        'Precision': precision_col,
        'Recall': recall_col,
        'F1 Score': 'f1',
        'mAP@0.5': mAP50_col,
        'mAP@0.5:0.95': mAP50_95_col
    }
    
    for name, col in metrics.items():
        initial = df[col].iloc[0]
        final = df[col].iloc[-1]
        best = df[col].max()
        best_epoch = df[col].idxmax() + 1
        change = ((final - initial) / initial * 100) if initial > 0 else float('inf')
        
        print(f"{name:<20s} {initial:>10.4f} {final:>10.4f} {best:>10.4f} {best_epoch:>12d} {change:>+9.1f}%")
    
    # ==== F1-CONFIDENCE ====
    print(f"\n🎯 F1-CONFIDENCE CORRELATION:")
    print("-" * 80)
    correlation = df['f1'].corr(df[mAP50_col])
    print(f"Correlation Coefficient: {correlation:.4f}")
    
    if correlation > 0.7:
        print("✅ Strong positive correlation - Well-balanced model")
    elif correlation > 0.5:
        print("🟡 Moderate correlation - Good learning pattern")
    else:
        print("⚠️ Weak correlation - May need adjustment")
    
    # ==== BEST POINTS ====
    best_f1_epoch = df['f1'].idxmax() + 1
    best_map_epoch = df[mAP50_col].idxmax() + 1
    
    print(f"\n📊 BEST F1 SCORE (Epoch {best_f1_epoch}):")
    print(f"   F1:       {df['f1'].iloc[best_f1_epoch-1]:.4f}")
    print(f"   Precision: {df[precision_col].iloc[best_f1_epoch-1]:.4f}")
    print(f"   Recall:    {df[recall_col].iloc[best_f1_epoch-1]:.4f}")
    print(f"   mAP@0.5:   {df[mAP50_col].iloc[best_f1_epoch-1]:.4f}")
    
    print(f"\n📊 BEST CONFIDENCE (Epoch {best_map_epoch}):")
    print(f"   mAP@0.5:   {df[mAP50_col].iloc[best_map_epoch-1]:.4f}")
    print(f"   F1:        {df['f1'].iloc[best_map_epoch-1]:.4f}")
    print(f"   Precision: {df[precision_col].iloc[best_map_epoch-1]:.4f}")
    print(f"   Recall:    {df[recall_col].iloc[best_map_epoch-1]:.4f}")
    
    # ==== RATING ====
    final_f1 = df['f1'].iloc[-1]
    final_map = df[mAP50_col].iloc[-1]
    best_f1 = df['f1'].max()
    best_map = df[mAP50_col].max()
    
    print(f"\n🏆 FINAL ASSESSMENT:")
    print("-" * 80)
    
    # Use best values for rating
    if best_f1 >= 0.7 and best_map >= 0.6:
        rating = "🏆 EXCELLENT - Production Ready"
    elif best_f1 >= 0.5 and best_map >= 0.4:
        rating = "✅ GOOD - Acceptable Performance"
    elif best_f1 >= 0.3 and best_map >= 0.2:
        rating = "🟡 MODERATE - Promising, needs more training"
    else:
        rating = "🔶 DEVELOPING - Continue training"
    
    print(f"Rating: {rating}")
    print(f"Best F1 Score:  {best_f1:.4f} (Epoch {best_f1_epoch})")
    print(f"Best mAP@0.5:   {best_map:.4f} (Epoch {best_map_epoch})")
    print(f"Final F1 Score: {final_f1:.4f}")
    print(f"Final mAP@0.5:  {final_map:.4f}")
    
    return {
        'name': dataset_name,
        'epochs': len(df),
        'best_f1': best_f1,
        'best_map': best_map,
        'final_f1': final_f1,
        'final_map': final_map,
        'loss_reduction': total_reduction
    }

def main():
    """Main function"""
    
    print("\n" + "="*80)
    print("🚀 YOLOv8 TRAINING IMPROVEMENTS ANALYSIS")
    print("="*80)
    print("Analyzing YOLOv8 models trained on crack_augmented and spalling_rebar datasets\n")
    
    # Training results
    training_results = [
        ('Spalling & Exposed Rebar - Segmentation', 'runs/segment/yolov8_spalling_rebar_segmentation/results.csv'),
        ('Spalling & Exposed Rebar - Final v2', 'runs/segment/yolov8_spalling_rebar_final2/results.csv'),
        ('Spalling & Exposed Rebar - Final v3', 'runs/segment/yolov8_spalling_rebar_final3/results.csv'),
    ]
    
    summaries = []
    
    for dataset_name, path in training_results:
        result = analyze_training(path, dataset_name)
        if result:
            summaries.append(result)
    
    # ==== COMPARISON SUMMARY ====
    if len(summaries) > 1:
        print(f"\n\n{'='*80}")
        print("📊 COMPARISON SUMMARY - ALL TRAINING RUNS")
        print("="*80)
        print(f"\n{'Dataset':<45s} {'Epochs':>8s} {'Best F1':>10s} {'Best mAP':>10s} {'Loss ↓':>10s}")
        print("-" * 80)
        
        for s in summaries:
            print(f"{s['name']:<45s} {s['epochs']:>8d} {s['best_f1']:>10.4f} {s['best_map']:>10.4f} {s['loss_reduction']:>9.1f}%")
        
        # Find best
        best_f1_model = max(summaries, key=lambda x: x['best_f1'])
        best_map_model = max(summaries, key=lambda x: x['best_map'])
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Best F1 Score:  {best_f1_model['name']} ({best_f1_model['best_f1']:.4f})")
        print(f"   Best mAP@0.5:   {best_map_model['name']} ({best_map_model['best_map']:.4f})")
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\n💡 RECOMMENDATIONS:")
    print("   • Review best performing epochs for early stopping")
    print("   • Check if model needs more epochs for convergence")
    print("   • Consider data augmentation if performance plateaus")
    print("   • Use best checkpoint (not final) for inference")

if __name__ == "__main__":
    main()


