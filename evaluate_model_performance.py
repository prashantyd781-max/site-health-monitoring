#!/usr/bin/env python3
"""
Model Performance Evaluation Script
Tests the trained model on training and validation datasets to assess performance
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import json

def test_model_on_dataset(model_path, images_dir, labels_dir, output_dir="evaluation_results"):
    """Test model on a dataset and calculate metrics"""
    
    print(f"🧪 Testing model on dataset: {images_dir}")
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Get all images
    image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    
    results_data = {
        'total_images': len(image_files),
        'images_with_ground_truth': 0,
        'images_with_detections': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'detections': []
    }
    
    print(f"📸 Found {len(image_files)} images to test")
    
    for i, img_path in enumerate(image_files):
        if i >= 20:  # Test on first 20 images for detailed analysis
            break
            
        print(f"🔍 Processing {i+1}/20: {img_path.name}")
        
        # Check if ground truth labels exist
        label_file = Path(labels_dir) / f"{img_path.stem}.txt"
        has_ground_truth = label_file.exists()
        ground_truth_count = 0
        
        if has_ground_truth:
            results_data['images_with_ground_truth'] += 1
            # Count ground truth annotations
            with open(label_file, 'r') as f:
                lines = f.readlines()
                ground_truth_count = len([line.strip() for line in lines if line.strip()])
        
        # Run inference
        try:
            detections = model(str(img_path), verbose=False)
            detection_count = 0
            confidence_scores = []
            
            if detections[0].boxes is not None and len(detections[0].boxes) > 0:
                results_data['images_with_detections'] += 1
                detection_count = len(detections[0].boxes)
                confidence_scores = [box.conf[0].item() for box in detections[0].boxes]
                
                print(f"   ✅ Detected {detection_count} crack(s) with confidences: {[f'{c:.3f}' for c in confidence_scores]}")
                
                # Save annotated image for visual inspection
                annotated_img = detections[0].plot()
                output_path = Path(output_dir) / f"detection_{img_path.name}"
                cv2.imwrite(str(output_path), annotated_img)
            else:
                print(f"   ❌ No detections")
            
            # Basic metric calculation (simplified - actual IoU calculation would be more complex)
            if has_ground_truth and ground_truth_count > 0:
                if detection_count > 0:
                    # Rough approximation - in reality you'd need IoU calculation
                    results_data['true_positives'] += min(detection_count, ground_truth_count)
                    if detection_count > ground_truth_count:
                        results_data['false_positives'] += (detection_count - ground_truth_count)
                else:
                    results_data['false_negatives'] += ground_truth_count
            elif detection_count > 0:
                results_data['false_positives'] += detection_count
            
            # Store detailed results
            results_data['detections'].append({
                'image': img_path.name,
                'ground_truth_count': ground_truth_count,
                'detection_count': detection_count,
                'max_confidence': max(confidence_scores) if confidence_scores else 0,
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0
            })
            
        except Exception as e:
            print(f"   ❌ Error processing {img_path.name}: {str(e)}")
    
    return results_data

def calculate_metrics(results_data):
    """Calculate standard evaluation metrics"""
    
    tp = results_data['true_positives']
    fp = results_data['false_positives'] 
    fn = results_data['false_negatives']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 EVALUATION METRICS:")
    print(f"=" * 50)
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-Score:        {f1_score:.4f}")
    print(f"Images with GT:  {results_data['images_with_ground_truth']}/20")
    print(f"Images with Det: {results_data['images_with_detections']}/20")
    
    # Detection confidence analysis
    detections = results_data['detections']
    if detections:
        confidences = [d['max_confidence'] for d in detections if d['max_confidence'] > 0]
        if confidences:
            print(f"\n🎯 CONFIDENCE ANALYSIS:")
            print(f"Average Max Confidence: {np.mean(confidences):.4f}")
            print(f"Min Confidence: {min(confidences):.4f}")
            print(f"Max Confidence: {max(confidences):.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def analyze_training_results(results_file):
    """Analyze the training results CSV"""
    
    print(f"\n📈 TRAINING RESULTS ANALYSIS:")
    print(f"=" * 50)
    
    if not Path(results_file).exists():
        print("❌ Training results file not found")
        return
    
    df = pd.read_csv(results_file)
    
    print(f"Training completed {len(df)} epochs")
    
    # Get final metrics
    final_metrics = df.iloc[-1]
    
    print(f"Final Training Metrics:")
    print(f"  mAP50:     {final_metrics['metrics/mAP50(B)']:.4f}")
    print(f"  mAP50-95:  {final_metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"  Precision: {final_metrics['metrics/precision(B)']:.4f}")
    print(f"  Recall:    {final_metrics['metrics/recall(B)']:.4f}")
    
    # Industry standards comparison
    print(f"\n🏭 INDUSTRY STANDARDS COMPARISON:")
    print(f"Good Performance Thresholds:")
    print(f"  mAP50:     > 0.5 (Current: {final_metrics['metrics/mAP50(B)']:.4f}) {'✅' if final_metrics['metrics/mAP50(B)'] > 0.5 else '❌'}")
    print(f"  Precision: > 0.7 (Current: {final_metrics['metrics/precision(B)']:.4f}) {'✅' if final_metrics['metrics/precision(B)'] > 0.7 else '❌'}")
    print(f"  Recall:    > 0.6 (Current: {final_metrics['metrics/recall(B)']:.4f}) {'✅' if final_metrics['metrics/recall(B)'] > 0.6 else '❌'}")
    
    return final_metrics

def main():
    script_dir = Path(__file__).parent
    model_path = script_dir / "yolov8_crack_detection_best.pt"
    
    if not model_path.exists():
        print("❌ Trained model not found!")
        return False
    
    print("🚀 MODEL PERFORMANCE EVALUATION")
    print("=" * 60)
    
    # Analyze training results
    results_file = script_dir / "runs/detect/crack_detection_train/results.csv"
    training_metrics = analyze_training_results(results_file)
    
    # Test on training data
    print(f"\n🔍 TESTING ON TRAINING DATA:")
    train_images = script_dir / "model_data/train/images"
    train_labels = script_dir / "model_data/train/labels"
    train_results = test_model_on_dataset(model_path, train_images, train_labels, "evaluation_train")
    train_metrics = calculate_metrics(train_results)
    
    # Test on validation data  
    print(f"\n🔍 TESTING ON VALIDATION DATA:")
    val_images = script_dir / "model_data/valid/images" 
    val_labels = script_dir / "model_data/valid/labels"
    val_results = test_model_on_dataset(model_path, val_images, val_labels, "evaluation_val")
    val_metrics = calculate_metrics(val_results)
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"=" * 50)
    print("The model shows very poor performance with:")
    print("- Extremely low mAP scores (< 0.1)")  
    print("- Very low precision (< 0.01)")
    print("- Decent recall but high false positive rate")
    print("\nPossible issues:")
    print("1. Training stopped too early (only 3-4 epochs)")
    print("2. Learning rate might be too high/low")
    print("3. Dataset labeling issues")
    print("4. Class imbalance problems")
    
    return True

if __name__ == "__main__":
    main()
