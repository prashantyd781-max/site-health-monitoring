#!/usr/bin/env python3
"""
Crack Detection Demo Script
Simple demonstration of how to use the trained YOLOv8 crack detection model
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

def detect_cracks(image_path, model_path="yolov8_crack_detection_best.pt", output_dir="detection_results"):
    """
    Detect cracks in an image using the trained model
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained model
        output_dir: Directory to save results
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load the trained model
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"🔍 Analyzing image: {image_path}")
    
    try:
        # Run detection
        results = model(image_path)
        
        # Process results
        for i, result in enumerate(results):
            # Get the original image
            img = result.orig_img
            
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"🚨 Found {len(result.boxes)} crack(s)!")
                
                # Get detection details
                for box in result.boxes:
                    confidence = box.conf[0].item()
                    print(f"   - Confidence: {confidence:.2f}")
                
                # Save annotated image
                annotated_img = result.plot()
                output_path = Path(output_dir) / f"detected_{Path(image_path).name}"
                cv2.imwrite(str(output_path), annotated_img)
                print(f"💾 Saved annotated image: {output_path}")
                
            else:
                print("✅ No cracks detected in this image")
                # Save original image for reference
                output_path = Path(output_dir) / f"no_cracks_{Path(image_path).name}"
                cv2.imwrite(str(output_path), img)
                print(f"💾 Saved reference image: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 demo_crack_detection.py <image_path>")
        print("Example: python3 demo_crack_detection.py brick-veg.jpg")
        return
    
    image_path = sys.argv[1]
    
    print("🧪 YOLOv8 Crack Detection Demo")
    print("=" * 40)
    
    success = detect_cracks(image_path)
    
    if success:
        print("\n🎉 Detection completed successfully!")
        print("📁 Check the 'detection_results' folder for output images")
    else:
        print("\n❌ Detection failed!")

if __name__ == "__main__":
    main()
