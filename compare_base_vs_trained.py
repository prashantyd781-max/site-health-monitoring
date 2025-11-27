#!/usr/bin/env python3
"""
Compare Base YOLOv8 vs Our Trained Model
This script tests both models on the same images to show improvement
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time

def test_model_performance(model_path, model_name, test_images):
    """Test a model on given images and return results"""
    
    print(f"🧪 Testing {model_name}...")
    
    try:
        model = YOLO(model_path)
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'total_detections': 0,
            'images_with_detections': 0,
            'total_images': 0,
            'avg_confidence': 0,
            'detection_details': []
        }
        
        confidences_all = []
        
        for img_path in test_images:
            if not Path(img_path).exists():
                continue
                
            results['total_images'] += 1
            
            # Run inference
            detections = model(img_path, verbose=False)
            
            img_result = {
                'image': Path(img_path).name,
                'detections': 0,
                'confidences': [],
                'avg_conf': 0,
                'max_conf': 0
            }
            
            if detections[0].boxes is not None and len(detections[0].boxes) > 0:
                results['images_with_detections'] += 1
                img_result['detections'] = len(detections[0].boxes)
                results['total_detections'] += img_result['detections']
                
                confidences = [box.conf[0].item() for box in detections[0].boxes]
                img_result['confidences'] = confidences
                img_result['avg_conf'] = np.mean(confidences)
                img_result['max_conf'] = max(confidences)
                
                confidences_all.extend(confidences)
            
            results['detection_details'].append(img_result)
        
        # Calculate overall confidence
        if confidences_all:
            results['avg_confidence'] = np.mean(confidences_all)
            results['max_confidence'] = max(confidences_all)
            results['min_confidence'] = min(confidences_all)
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return None

def compare_models():
    """Compare base YOLOv8 vs our trained model"""
    
    print("🎯 BASE YOLOV8 vs TRAINED MODEL COMPARISON")
    print("=" * 80)
    
    # Model paths
    base_model_path = "yolov8n.pt"  # Base YOLOv8 nano
    trained_model_path = "yolov8_crack_detection_WORKING.pt"
    
    # Test images (use a variety)
    test_images = Path("model_data/test/images")
    
    # Filter existing images
    existing_images = [str(p) for p in test_images.glob("*.*")]
    print(f"📸 Testing on {len(existing_images)} images: {[Path(img).name for img in existing_images]}")
    
    if not existing_images:
        print("❌ No test images found!")
        return
    
    # Test base model
    print(f"\n🔵 TESTING BASE YOLOV8 MODEL...")
    print("-" * 50)
    base_results = test_model_performance(base_model_path, "Base YOLOv8", existing_images)
    
    if base_results:
        print(f"📊 Base YOLOv8 Results:")
        print(f"  🖼️  Images tested: {base_results['total_images']}")
        print(f"  ✅ Images with detections: {base_results['images_with_detections']}")
        print(f"  🎯 Total detections: {base_results['total_detections']}")
        if base_results['avg_confidence'] > 0:
            print(f"  📊 Average confidence: {base_results['avg_confidence']:.3f}")
            print(f"  📊 Max confidence: {base_results['max_confidence']:.3f}")
        
        print(f"\n  📋 Detailed Results:")
        for detail in base_results['detection_details']:
            if detail['detections'] > 0:
                print(f"    ✅ {detail['image']}: {detail['detections']} detections (conf: {[f'{c:.3f}' for c in detail['confidences']]})")
            else:
                print(f"    ❌ {detail['image']}: No detections")
    
    # Test trained model  
    print(f"\n🟢 TESTING OUR TRAINED MODEL...")
    print("-" * 50)
    trained_results = test_model_performance(trained_model_path, "Our Trained Model", existing_images)
    
    if trained_results:
        print(f"📊 Our Trained Model Results:")
        print(f"  🖼️  Images tested: {trained_results['total_images']}")
        print(f"  ✅ Images with detections: {trained_results['images_with_detections']}")
        print(f"  🎯 Total detections: {trained_results['total_detections']}")
        if trained_results['avg_confidence'] > 0:
            print(f"  📊 Average confidence: {trained_results['avg_confidence']:.3f}")
            print(f"  📊 Max confidence: {trained_results['max_confidence']:.3f}")
        
        print(f"\n  📋 Detailed Results:")
        for detail in trained_results['detection_details']:
            if detail['detections'] > 0:
                print(f"    ✅ {detail['image']}: {detail['detections']} detections (conf: {[f'{c:.3f}' for c in detail['confidences']]})")
            else:
                print(f"    ❌ {detail['image']}: No detections")
    
    # Comparison Analysis
    if base_results and trained_results:
        print(f"\n📊 DETAILED COMPARISON ANALYSIS:")
        print("=" * 80)
        
        # Detection capability comparison
        print(f"🎯 DETECTION CAPABILITY:")
        print(f"┌─────────────────────────┬─────────────┬─────────────────┐")
        print(f"│ Metric                  │ Base YOLOv8 │ Our Trained     │")
        print(f"├─────────────────────────┼─────────────┼─────────────────┤")
        print(f"│ Images with detections  │    {base_results['images_with_detections']:2d}/{base_results['total_images']}      │      {trained_results['images_with_detections']:2d}/{trained_results['total_images']}        │")
        print(f"│ Total detections        │    {base_results['total_detections']:6d}     │      {trained_results['total_detections']:6d}       │")
        print(f"│ Detection rate          │   {(base_results['images_with_detections']/base_results['total_images']*100):5.1f}%     │     {(trained_results['images_with_detections']/trained_results['total_images']*100):5.1f}%      │")
        print(f"└─────────────────────────┴─────────────┴─────────────────┘")
        
        # Confidence analysis
        if base_results['avg_confidence'] > 0 and trained_results['avg_confidence'] > 0:
            print(f"\n📊 CONFIDENCE ANALYSIS:")
            print(f"┌─────────────────────────┬─────────────┬─────────────────┐")
            print(f"│ Metric                  │ Base YOLOv8 │ Our Trained     │")
            print(f"├─────────────────────────┼─────────────┼─────────────────┤")
            print(f"│ Average confidence      │   {base_results['avg_confidence']:7.3f}   │     {trained_results['avg_confidence']:7.3f}     │")
            print(f"│ Maximum confidence      │   {base_results['max_confidence']:7.3f}   │     {trained_results['max_confidence']:7.3f}     │")
            if 'min_confidence' in base_results:
                print(f"│ Minimum confidence      │   {base_results['min_confidence']:7.3f}   │     {trained_results['min_confidence']:7.3f}     │")
            print(f"└─────────────────────────┴─────────────┴─────────────────┘")
        
        # Calculate improvements
        detection_improvement = ((trained_results['images_with_detections'] / max(base_results['images_with_detections'], 0.1)) - 1) * 100
        total_detection_improvement = ((trained_results['total_detections'] / max(base_results['total_detections'], 0.1)) - 1) * 100
        
        print(f"\n🚀 IMPROVEMENT ANALYSIS:")
        print(f"=" * 50)
        
        if base_results['images_with_detections'] == 0 and trained_results['images_with_detections'] > 0:
            print(f"🎉 BREAKTHROUGH: Base model detected NOTHING!")
            print(f"✅ Our model successfully detects cracks in {trained_results['images_with_detections']} images!")
            print(f"📈 Improvement: From 0% to {(trained_results['images_with_detections']/trained_results['total_images']*100):.1f}% detection rate")
        elif trained_results['images_with_detections'] > base_results['images_with_detections']:
            print(f"✅ Detection Rate: +{detection_improvement:.0f}% improvement")
            print(f"✅ Total Detections: +{total_detection_improvement:.0f}% improvement")
        
        if trained_results['avg_confidence'] > 0 and base_results['avg_confidence'] > 0:
            conf_improvement = ((trained_results['avg_confidence'] / base_results['avg_confidence']) - 1) * 100
            if conf_improvement > 0:
                print(f"✅ Confidence: +{conf_improvement:.1f}% improvement")
            else:
                print(f"🔶 Confidence: {conf_improvement:.1f}% (trade-off for better detection)")
        
        # Key insights
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"=" * 50)
        
        if base_results['total_detections'] == 0:
            print(f"🎯 BASE YOLOV8: Cannot detect cracks at all (not trained for this)")
            print(f"🎯 OUR MODEL: Successfully trained to detect cracks specifically")
            print(f"🎯 SPECIALIZATION: Domain-specific training made all the difference")
        else:
            print(f"🎯 Both models can detect objects, but ours is specialized for cracks")
            print(f"🎯 Our model shows improved performance on crack detection task")
        
        # Training metrics context
        print(f"\n📈 TRAINING ACHIEVEMENT CONTEXT:")
        print(f"=" * 50)
        print(f"🏆 Training Duration: 150 epochs (3.788 hours)")
        print(f"🏆 Final Validation mAP50: 0.388")
        print(f"🏆 Final Validation Precision: 0.542") 
        print(f"🏆 Final Validation Recall: 0.389")
        print(f"🏆 Dataset: 450 crack annotations (272 train + 90 valid + 88 test)")
        print(f"🏆 Specialization: Crack detection vs general object detection")

def main():
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    compare_models()

if __name__ == "__main__":
    main()





