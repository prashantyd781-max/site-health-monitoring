#!/usr/bin/env python3
"""
Test Spalling & Exposed Rebar Detection Model
Usage: python test_spalling_rebar_model.py <image_path>
"""

from ultralytics import YOLO
import sys
import os
from pathlib import Path

def detect_defects(model_path='best.pt', image_path=None, conf_threshold=0.25, save_output=True):
    """
    Detect spalling and exposed rebar in an image
    
    Args:
        model_path: Path to trained model (.pt file)
        image_path: Path to input image
        conf_threshold: Confidence threshold (0-1)
        save_output: Whether to save annotated image
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("💡 Make sure 'best.pt' is in the current directory")
        return
    
    # Check if image exists
    if not image_path or not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("💡 Usage: python test_spalling_rebar_model.py <image_path>")
        return
    
    print(f"🔧 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"🔍 Analyzing image: {image_path}")
    print(f"⚙️  Confidence threshold: {conf_threshold}\n")
    
    # Run inference
    results = model(image_path, conf=conf_threshold, verbose=False)
    
    # Get results
    result = results[0]
    boxes = result.boxes
    
    print("=" * 80)
    print("📊 DETECTION RESULTS")
    print("=" * 80 + "\n")
    
    if len(boxes) > 0:
        print(f"✅ Detected {len(boxes)} defect(s):\n")
        
        # Count by class
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Count
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            # Print detection
            print(f"   • {cls_name:20s} - Confidence: {conf:.3f} ({conf*100:.1f}%)")
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"     Location: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        # Summary
        print("\n" + "-" * 80)
        print("📈 Summary:")
        for cls_name, count in class_counts.items():
            print(f"   {cls_name}: {count} instance(s)")
        
    else:
        print("ℹ️  No defects detected (above confidence threshold)")
        print(f"💡 Try lowering confidence threshold (current: {conf_threshold})")
    
    print("\n" + "=" * 80)
    
    # Save annotated image
    if save_output:
        output_path = f"detected_{Path(image_path).name}"
        result.save(output_path)
        print(f"\n💾 Saved annotated image: {output_path}")
    
    # Display image (if in interactive environment)
    try:
        result.show()
    except:
        pass
    
    return results


def batch_detect(model_path='best.pt', images_folder=None, conf_threshold=0.25):
    """
    Detect defects in multiple images from a folder
    
    Args:
        model_path: Path to trained model
        images_folder: Folder containing images
        conf_threshold: Confidence threshold
    """
    
    if not images_folder or not os.path.exists(images_folder):
        print(f"❌ Folder not found: {images_folder}")
        return
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in image_extensions:
        images.extend(Path(images_folder).glob(ext))
    
    if not images:
        print(f"❌ No images found in: {images_folder}")
        return
    
    print(f"🔧 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"🔍 Processing {len(images)} images from: {images_folder}\n")
    
    # Process each image
    results_summary = []
    
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing: {img_path.name}")
        
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        num_detections = len(results[0].boxes)
        
        # Save result
        output_path = f"detected_{img_path.name}"
        results[0].save(output_path)
        
        # Count by class
        class_counts = {}
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        summary_text = ", ".join([f"{name}: {count}" for name, count in class_counts.items()]) if class_counts else "No defects"
        print(f"   {summary_text}\n")
        
        results_summary.append({
            'image': img_path.name,
            'detections': num_detections,
            'classes': class_counts
        })
    
    # Print overall summary
    print("=" * 80)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {len(images)}")
    
    total_detections = sum(r['detections'] for r in results_summary)
    images_with_defects = sum(1 for r in results_summary if r['detections'] > 0)
    
    print(f"Images with defects: {images_with_defects}/{len(images)}")
    print(f"Total defects detected: {total_detections}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect spalling and exposed rebar in images')
    parser.add_argument('image', nargs='?', help='Path to input image or folder')
    parser.add_argument('--model', default='best.pt', help='Path to model file (default: best.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--batch', action='store_true', help='Process all images in folder')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("🏗️  SPALLING & EXPOSED REBAR DETECTION")
    print("=" * 80 + "\n")
    
    if args.image:
        if args.batch:
            batch_detect(args.model, args.image, args.conf)
        else:
            detect_defects(args.model, args.image, args.conf)
    else:
        print("Usage Examples:")
        print("  Single image:  python test_spalling_rebar_model.py image.jpg")
        print("  Batch folder:  python test_spalling_rebar_model.py folder/ --batch")
        print("  Custom conf:   python test_spalling_rebar_model.py image.jpg --conf 0.5")
        print("\nOptions:")
        print("  --model MODEL  Path to model file (default: best.pt)")
        print("  --conf CONF    Confidence threshold 0-1 (default: 0.25)")
        print("  --batch        Process all images in folder")


