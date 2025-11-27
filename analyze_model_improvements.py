#!/usr/bin/env python3
"""
Comprehensive Model Improvement Analysis
Analyzes losses, F1 confidence, and performance metrics across training runs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

class ModelAnalyzer:
    def __init__(self):
        self.training_runs = {
            'YOLOv11': 'runs/detect/yolo11_crack_detection_train/results.csv',
            'YOLOv8': 'runs/detect/yolov8_crack_detection/results.csv',
            'Fixed Training': 'runs/detect/crack_detection_fixed/results.csv',
            'Original Training': 'runs/detect/crack_detection_train/results.csv',
            'Train V3': 'runs/detect/train3/results.csv',
            'Train V1': 'runs/detect/train/results.csv',
        }
        self.colors = {
            'YOLOv11': '#FF6B6B',
            'YOLOv8': '#4ECDC4',
            'Fixed Training': '#45B7D1',
            'Original Training': '#96CEB4',
            'Train V3': '#FFEAA7',
            'Train V1': '#DFE6E9',
        }
    
    def load_data(self):
        """Load all available training data"""
        data = {}
        print("📦 Loading Training Data")
        print("=" * 80)
        
        for name, path in self.training_runs.items():
            path = Path(path)
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    data[name] = df
                    print(f"✅ {name:20s} - {len(df):3d} epochs")
                except Exception as e:
                    print(f"❌ {name:20s} - Failed: {e}")
        
        print("=" * 80 + "\n")
        return data
    
    def analyze_losses(self, data):
        """Detailed loss analysis"""
        print("\n📉 LOSS ANALYSIS")
        print("=" * 80)
        
        for name, df in data.items():
            if len(df) == 0:
                continue
                
            print(f"\n🔹 {name}")
            print("-" * 60)
            
            # Initial epoch
            first = df.iloc[0]
            # Final epoch
            last = df.iloc[-1]
            
            # Training losses
            print(f"  📊 Training Losses:")
            if 'train/box_loss' in df.columns:
                print(f"     Box Loss:      {first['train/box_loss']:.4f} → {last['train/box_loss']:.4f} "
                      f"({((last['train/box_loss']/first['train/box_loss']-1)*100):+.1f}%)")
            if 'train/cls_loss' in df.columns:
                print(f"     Class Loss:    {first['train/cls_loss']:.4f} → {last['train/cls_loss']:.4f} "
                      f"({((last['train/cls_loss']/first['train/cls_loss']-1)*100):+.1f}%)")
            if 'train/dfl_loss' in df.columns:
                print(f"     DFL Loss:      {first['train/dfl_loss']:.4f} → {last['train/dfl_loss']:.4f} "
                      f"({((last['train/dfl_loss']/first['train/dfl_loss']-1)*100):+.1f}%)")
            
            # Validation losses
            print(f"  📊 Validation Losses:")
            if 'val/box_loss' in df.columns:
                print(f"     Box Loss:      {first['val/box_loss']:.4f} → {last['val/box_loss']:.4f} "
                      f"({((last['val/box_loss']/first['val/box_loss']-1)*100):+.1f}%)")
            if 'val/cls_loss' in df.columns:
                print(f"     Class Loss:    {first['val/cls_loss']:.4f} → {last['val/cls_loss']:.4f} "
                      f"({((last['val/cls_loss']/first['val/cls_loss']-1)*100):+.1f}%)")
            if 'val/dfl_loss' in df.columns:
                print(f"     DFL Loss:      {first['val/dfl_loss']:.4f} → {last['val/dfl_loss']:.4f} "
                      f"({((last['val/dfl_loss']/first['val/dfl_loss']-1)*100):+.1f}%)")
        
        print("=" * 80)
    
    def analyze_metrics(self, data):
        """Analyze precision, recall, F1, mAP"""
        print("\n📈 PERFORMANCE METRICS ANALYSIS")
        print("=" * 80)
        
        comparison = []
        
        for name, df in data.items():
            if len(df) == 0:
                continue
                
            last = df.iloc[-1]
            
            # Determine metric column names (B for box, M for mask/segmentation)
            metric_suffix = '(B)' if 'metrics/precision(B)' in df.columns else '(M)'
            
            metrics = {
                'Model': name,
                'Epochs': len(df),
                'Precision': last.get(f'metrics/precision{metric_suffix}', 0),
                'Recall': last.get(f'metrics/recall{metric_suffix}', 0),
                'mAP50': last.get(f'metrics/mAP50{metric_suffix}', 0),
                'mAP50-95': last.get(f'metrics/mAP50-95{metric_suffix}', 0),
            }
            
            # Calculate F1 Score
            if metrics['Precision'] > 0 and metrics['Recall'] > 0:
                metrics['F1'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall'])
            else:
                metrics['F1'] = 0
            
            comparison.append(metrics)
            
            print(f"\n🔹 {name}")
            print(f"   Precision:    {metrics['Precision']:.4f}")
            print(f"   Recall:       {metrics['Recall']:.4f}")
            print(f"   F1 Score:     {metrics['F1']:.4f}")
            print(f"   mAP@0.5:      {metrics['mAP50']:.4f}")
            print(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            print(f"   Epochs:       {metrics['Epochs']}")
        
        print("\n" + "=" * 80)
        print("📊 COMPARISON TABLE")
        print("=" * 80)
        
        # Sort by F1 score
        comparison.sort(key=lambda x: x['F1'], reverse=True)
        
        print(f"\n{'Rank':<6}{'Model':<25}{'F1':>8}{'Precision':>11}{'Recall':>9}{'mAP50':>9}")
        print("-" * 80)
        for idx, m in enumerate(comparison, 1):
            print(f"{idx:<6}{m['Model']:<25}{m['F1']:>8.4f}{m['Precision']:>11.4f}{m['Recall']:>9.4f}{m['mAP50']:>9.4f}")
        
        print("=" * 80)
        
        # Best model
        if comparison:
            best = comparison[0]
            print(f"\n🏆 BEST MODEL: {best['Model']}")
            print(f"   F1 Score: {best['F1']:.4f}")
            print(f"   mAP@0.5:  {best['mAP50']:.4f}")
        
        return comparison
    
    def plot_loss_comparison(self, data):
        """Create comprehensive loss comparison plots"""
        print("\n📊 Generating Loss Comparison Plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Training - Loss Comparison', fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Total Combined Loss
        ax1.set_title('Total Combined Loss (Train + Val)', fontsize=13, fontweight='bold')
        for name, df in data.items():
            if len(df) == 0:
                continue
            
            epochs = range(1, len(df) + 1)
            
            # Calculate total loss
            train_loss = None
            val_loss = None
            
            if 'train/box_loss' in df.columns and 'train/cls_loss' in df.columns:
                train_loss = df['train/box_loss'] + df['train/cls_loss']
                if 'train/dfl_loss' in df.columns:
                    train_loss += df['train/dfl_loss']
            
            if 'val/box_loss' in df.columns and 'val/cls_loss' in df.columns:
                val_loss = df['val/box_loss'] + df['val/cls_loss']
                if 'val/dfl_loss' in df.columns:
                    val_loss += df['val/dfl_loss']
            
            color = self.colors.get(name, '#000000')
            
            if train_loss is not None:
                ax1.plot(epochs, train_loss, label=f'{name} (Train)', 
                        color=color, linewidth=2, linestyle='-', alpha=0.8)
            if val_loss is not None:
                ax1.plot(epochs, val_loss, color=color, linewidth=2, 
                        linestyle='--', alpha=0.6)
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Total Loss', fontsize=11)
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Box Loss
        ax2.set_title('Box Loss', fontsize=13, fontweight='bold')
        for name, df in data.items():
            if len(df) == 0 or 'train/box_loss' not in df.columns:
                continue
            
            epochs = range(1, len(df) + 1)
            color = self.colors.get(name, '#000000')
            
            ax2.plot(epochs, df['train/box_loss'], label=f'{name} (Train)', 
                    color=color, linewidth=2, linestyle='-', alpha=0.8)
            if 'val/box_loss' in df.columns:
                ax2.plot(epochs, df['val/box_loss'], color=color, 
                        linewidth=2, linestyle='--', alpha=0.6)
        
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Box Loss', fontsize=11)
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. Classification Loss
        ax3.set_title('Classification Loss', fontsize=13, fontweight='bold')
        for name, df in data.items():
            if len(df) == 0 or 'train/cls_loss' not in df.columns:
                continue
            
            epochs = range(1, len(df) + 1)
            color = self.colors.get(name, '#000000')
            
            ax3.plot(epochs, df['train/cls_loss'], label=f'{name} (Train)', 
                    color=color, linewidth=2, linestyle='-', alpha=0.8)
            if 'val/cls_loss' in df.columns:
                ax3.plot(epochs, df['val/cls_loss'], color=color, 
                        linewidth=2, linestyle='--', alpha=0.6)
        
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Classification Loss', fontsize=11)
        ax3.legend(fontsize=8, loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. DFL Loss
        ax4.set_title('DFL Loss', fontsize=13, fontweight='bold')
        for name, df in data.items():
            if len(df) == 0 or 'train/dfl_loss' not in df.columns:
                continue
            
            epochs = range(1, len(df) + 1)
            color = self.colors.get(name, '#000000')
            
            ax4.plot(epochs, df['train/dfl_loss'], label=f'{name} (Train)', 
                    color=color, linewidth=2, linestyle='-', alpha=0.8)
            if 'val/dfl_loss' in df.columns:
                ax4.plot(epochs, df['val/dfl_loss'], color=color, 
                        linewidth=2, linestyle='--', alpha=0.6)
        
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('DFL Loss', fontsize=11)
        ax4.legend(fontsize=8, loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = 'model_loss_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        
        return fig
    
    def plot_metrics_comparison(self, data):
        """Create performance metrics comparison plots"""
        print("📊 Generating Metrics Comparison Plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Training - Performance Metrics Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        for name, df in data.items():
            if len(df) == 0:
                continue
            
            epochs = range(1, len(df) + 1)
            color = self.colors.get(name, '#000000')
            
            # Determine metric suffix
            metric_suffix = '(B)' if 'metrics/precision(B)' in df.columns else '(M)'
            
            # 1. Precision
            if f'metrics/precision{metric_suffix}' in df.columns:
                ax1.plot(epochs, df[f'metrics/precision{metric_suffix}'], 
                        label=name, color=color, linewidth=2, alpha=0.8)
            
            # 2. Recall
            if f'metrics/recall{metric_suffix}' in df.columns:
                ax2.plot(epochs, df[f'metrics/recall{metric_suffix}'], 
                        label=name, color=color, linewidth=2, alpha=0.8)
            
            # 3. mAP@0.5
            if f'metrics/mAP50{metric_suffix}' in df.columns:
                ax3.plot(epochs, df[f'metrics/mAP50{metric_suffix}'], 
                        label=name, color=color, linewidth=2, alpha=0.8)
            
            # 4. F1 Score (calculated)
            if (f'metrics/precision{metric_suffix}' in df.columns and 
                f'metrics/recall{metric_suffix}' in df.columns):
                precision = df[f'metrics/precision{metric_suffix}']
                recall = df[f'metrics/recall{metric_suffix}']
                f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                ax4.plot(epochs, f1, label=name, color=color, linewidth=2, alpha=0.8)
        
        # Configure subplots
        ax1.set_title('Precision', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Precision', fontsize=11)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        ax2.set_title('Recall', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Recall', fontsize=11)
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        ax3.set_title('mAP@0.5', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('mAP@0.5', fontsize=11)
        ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        ax4.set_title('F1 Score', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('F1 Score', fontsize=11)
        ax4.legend(fontsize=9, loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        output_file = 'model_metrics_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        
        return fig
    
    def plot_f1_confidence_curves(self, data):
        """Plot F1-Confidence curves for all models"""
        print("📊 Generating F1-Confidence Curves...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('F1-Confidence Curves Comparison', fontsize=16, fontweight='bold')
        
        for name, df in data.items():
            if len(df) == 0:
                continue
            
            # Determine metric suffix
            metric_suffix = '(B)' if 'metrics/precision(B)' in df.columns else '(M)'
            
            if (f'metrics/precision{metric_suffix}' in df.columns and 
                f'metrics/recall{metric_suffix}' in df.columns and
                f'metrics/mAP50{metric_suffix}' in df.columns):
                
                precision = df[f'metrics/precision{metric_suffix}']
                recall = df[f'metrics/recall{metric_suffix}']
                confidence = df[f'metrics/mAP50{metric_suffix}']
                
                # Calculate F1 score
                f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                
                color = self.colors.get(name, '#000000')
                
                ax.plot(confidence, f1, label=name, color=color, 
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)
                
                # Mark best F1 point
                best_idx = f1.idxmax()
                ax.scatter(confidence.iloc[best_idx], f1.iloc[best_idx], 
                          color=color, s=150, marker='*', 
                          edgecolors='black', linewidths=1.5, zorder=10)
        
        ax.set_xlabel('Confidence (mAP@0.5)', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        output_file = 'f1_confidence_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        
        return fig
    
    def save_report(self, comparison):
        """Save analysis report to JSON and text file"""
        print("\n💾 Saving Analysis Report...")
        
        # JSON report
        json_file = 'model_analysis_report.json'
        with open(json_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"✅ Saved: {json_file}")
        
        # Text report
        txt_file = 'model_analysis_report.txt'
        with open(txt_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Sort by F1 score
            sorted_models = sorted(comparison, key=lambda x: x['F1'], reverse=True)
            
            for idx, model in enumerate(sorted_models, 1):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"RANK {idx}: {model['Model']}\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"  Training Duration: {model['Epochs']} epochs\n")
                f.write(f"  F1 Score:          {model['F1']:.4f}\n")
                f.write(f"  Precision:         {model['Precision']:.4f}\n")
                f.write(f"  Recall:            {model['Recall']:.4f}\n")
                f.write(f"  mAP@0.5:           {model['mAP50']:.4f}\n")
                f.write(f"  mAP@0.5:0.95:      {model['mAP50-95']:.4f}\n")
            
            if sorted_models:
                best = sorted_models[0]
                f.write(f"\n{'=' * 80}\n")
                f.write(f"🏆 BEST MODEL: {best['Model']}\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"  This model achieved the highest F1 score of {best['F1']:.4f}\n")
                f.write(f"  with mAP@0.5 of {best['mAP50']:.4f}\n")
        
        print(f"✅ Saved: {txt_file}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "🔬" * 40)
        print("COMPREHENSIVE MODEL IMPROVEMENT ANALYSIS")
        print("🔬" * 40 + "\n")
        
        # Load data
        data = self.load_data()
        
        if not data:
            print("❌ No training data found!")
            return
        
        # Analyze losses
        self.analyze_losses(data)
        
        # Analyze metrics
        comparison = self.analyze_metrics(data)
        
        # Generate plots
        print("\n📊 GENERATING VISUALIZATIONS")
        print("=" * 80)
        self.plot_loss_comparison(data)
        self.plot_metrics_comparison(data)
        self.plot_f1_confidence_curves(data)
        
        # Save report
        self.save_report(comparison)
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\n📁 Generated Files:")
        print("   📊 model_loss_comparison.png - Loss curves comparison")
        print("   📈 model_metrics_comparison.png - Performance metrics")
        print("   🎯 f1_confidence_comparison.png - F1-confidence curves")
        print("   📄 model_analysis_report.json - Detailed metrics (JSON)")
        print("   📝 model_analysis_report.txt - Human-readable report")
        print("\n💡 Check these files to see your model improvements!")


def main():
    analyzer = ModelAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

