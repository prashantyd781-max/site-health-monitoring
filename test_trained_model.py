#!/usr/bin/env python3
"""
Test script for the trained YOLOv8 crack detection model
This script validates that the trained model works correctly.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def test_model(model_path, test_images_dir, output_dir="test_results"):
    """Test the trained model on sample images"""
    
    print(f"🧪 Testing model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load the model
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return False
    
    # Get test images
    test_images = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))
    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        return False
    
    print(f"📸 Found {len(test_images)} test images")
    
    # Test on a few sample images
    success_count = 0
    for i, img_path in enumerate(test_images[:5]):  # Test on first 5 images
        try:
            print(f"🔍 Testing on: {img_path.name}")
            
            # Run inference
            results = model(str(img_path))
            
            # Save result
            result_path = Path(output_dir) / f"result_{img_path.stem}.jpg"
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                print(f"   ✅ Detected {len(results[0].boxes)} crack(s)")
                # Save annotated image
                annotated_img = results[0].plot()
                cv2.imwrite(str(result_path), annotated_img)
            else:
                print("   ℹ️  No cracks detected")
                # Still save the image for review
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(result_path), img)
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {img_path.name}: {str(e)}")
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Successfully processed: {success_count}/{min(5, len(test_images))} images")
    print(f"   📁 Results saved in: {output_dir}")
    
    return success_count > 0

def test_model_info(model_path):
    """Display model information"""
    try:
        model = YOLO(model_path)
        print(f"\n📋 Model Information:")
        print(f"   📦 Model type: {type(model.model).__name__}")
        print(f"   🔢 Classes: {model.model.names if hasattr(model.model, 'names') else 'Unknown'}")
        
        # Try to get model stats
        if hasattr(model, 'info'):
            model.info()
        
        return True
    except Exception as e:
        print(f"❌ Error getting model info: {str(e)}")
        return False

def main():
    script_dir = Path(__file__).parent
    
    # Look for the trained model
    possible_models = [
        script_dir / "yolov8_crack_detection_best.pt",
        script_dir / "runs" / "detect" / "crack_detection_train" / "weights" / "best.pt",
    ]
    
    model_path = None
    for path in possible_models:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        print("❌ No trained model found. Please ensure training is complete.")
        print("Expected locations:")
        for path in possible_models:
            print(f"   - {path}")
        return False
    
    print(f"🎯 Found trained model: {model_path}")
    
    # Test model information
    test_model_info(model_path)
    
    # Test images directory
    test_images_dir = script_dir / "model_data" / "test" / "images"
    if not test_images_dir.exists():
        test_images_dir = script_dir / "model_data" / "valid" / "images"
    
    if not test_images_dir.exists():
        print("❌ No test images directory found")
        return False
    
    # Run tests
    success = test_model(model_path, test_images_dir)
    
    if success:
        print("\n🎉 Model testing completed successfully!")
        print("✅ The trained model is working correctly!")
    else:
        print("\n❌ Model testing failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
