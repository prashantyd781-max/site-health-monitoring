# 🏗️ Spalling & Exposed Rebar Detection - Usage Guide

Congratulations on training your model! This guide will help you use it effectively.

---

## 📦 Files You Have

After training completion, you should have these files:

- **`best.pt`** ⭐ - Your trained model (main file to use)
- **`last.pt`** - Last epoch checkpoint (backup)
- **`spalling_rebar_training_results.zip`** - All training results and metrics

---

## 🚀 Quick Start Options

### Option 1: Command Line Testing (Easiest)

```bash
# Test on a single image
python test_spalling_rebar_model.py your_image.jpg

# Test on all images in a folder
python test_spalling_rebar_model.py images_folder/ --batch

# Adjust confidence threshold
python test_spalling_rebar_model.py image.jpg --conf 0.5
```

### Option 2: Web Interface (Most User-Friendly)

```bash
# Install Streamlit (if not already installed)
pip install streamlit

# Run the web app
streamlit run webapp_spalling_rebar.py
```

Then open your browser to `http://localhost:8501`

### Option 3: Python Code (Most Flexible)

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('best.pt')

# Detect on an image
results = model('path/to/image.jpg')

# Display results
results[0].show()

# Save annotated image
results[0].save('output.jpg')

# Get detection details
for box in results[0].boxes:
    cls_name = model.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    print(f"{cls_name}: {confidence:.2%}")
```

---

## 📊 Analyze Training Results

### Extract the ZIP File

Unzip `spalling_rebar_training_results.zip` to see:

```
spalling_rebar_training_results/
├── weights/
│   ├── best.pt              # Best model
│   └── last.pt              # Last epoch
├── results.png              # Training curves
├── results.csv              # Detailed metrics
├── confusion_matrix.png     # Confusion matrix
├── F1_curve.png            # F1-confidence curve
├── PR_curve.png            # Precision-Recall curve
├── P_curve.png             # Precision curve
├── R_curve.png             # Recall curve
└── val_batch*_pred.jpg     # Validation predictions
```

### Key Metrics to Check

Open `results.csv` to see:
- **mAP50** - Mean Average Precision at 50% IoU (higher is better)
- **Precision** - How many detections were correct
- **Recall** - How many actual defects were found
- **Loss values** - Should decrease over epochs

---

## 🎯 Real-World Usage

### 1. Inspect Construction Sites

```bash
python test_spalling_rebar_model.py construction_site.jpg
```

### 2. Batch Process Survey Images

```bash
python test_spalling_rebar_model.py survey_images/ --batch
```

### 3. Integrate into Your Application

```python
from ultralytics import YOLO

class DefectDetector:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
    
    def detect(self, image_path, conf=0.25):
        results = self.model(image_path, conf=conf)
        
        detections = []
        for box in results[0].boxes:
            detections.append({
                'class': self.model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
        
        return detections

# Use it
detector = DefectDetector()
defects = detector.detect('structure.jpg')
print(f"Found {len(defects)} defects")
```

---

## 🔧 Advanced Options

### Adjust Confidence Threshold

```python
# Lower threshold = more detections (but more false positives)
results = model('image.jpg', conf=0.15)  # Default is 0.25

# Higher threshold = fewer detections (but more accurate)
results = model('image.jpg', conf=0.50)
```

### Process Video

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Process video
results = model('video.mp4')

# Save annotated video
model('video.mp4', save=True)
```

### Export to Other Formats

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Export to ONNX (for deployment)
model.export(format='onnx')

# Export to TensorFlow Lite (for mobile)
model.export(format='tflite')

# Export to TensorRT (for NVIDIA GPUs)
model.export(format='engine')
```

---

## 📱 Deploy Options

### 1. **Desktop Application**
- Use the Streamlit web app (`webapp_spalling_rebar.py`)
- Package with PyInstaller for standalone executable

### 2. **Web Service**
- Deploy with FastAPI or Flask
- Host on cloud (AWS, Google Cloud, Azure)

### 3. **Mobile App**
- Export model to TFLite
- Integrate with Flutter or React Native

### 4. **Edge Device**
- Export to TensorRT for NVIDIA Jetson
- Use ONNX Runtime for Raspberry Pi

---

## 💡 Tips for Best Results

### Image Quality
- ✅ Use high-resolution images (640x640 or larger)
- ✅ Good lighting conditions
- ✅ Clear view of concrete surfaces
- ❌ Avoid blurry or low-light images

### Confidence Threshold
- **0.25** - Balanced (recommended for most cases)
- **0.15-0.20** - More detections (use if missing defects)
- **0.40-0.50** - Fewer false positives (use for critical decisions)

### Improving Model
- Collect more training images
- Include diverse conditions (lighting, angles, damage types)
- Retrain with additional data
- Try larger model (yolov8s.pt or yolov8m.pt)

---

## 🐛 Troubleshooting

### Model not found
```
❌ Error: Model file 'best.pt' not found
```
**Solution:** Make sure `best.pt` is in the same directory as your script

### Low confidence detections
```
ℹ️ No defects detected
```
**Solution:** 
- Lower confidence threshold: `--conf 0.15`
- Check if image contains actual defects
- Verify image quality

### Import errors
```
❌ ModuleNotFoundError: No module named 'ultralytics'
```
**Solution:** Install required packages
```bash
pip install ultralytics streamlit pillow
```

---

## 📚 Resources

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **YOLOv8 Guide**: https://github.com/ultralytics/ultralytics
- **Model Training**: Refer to `train_spalling_rebar_kaggle.ipynb`

---

## 🎯 Next Steps

1. ✅ **Test your model** on sample images
2. ✅ **Analyze training results** from the ZIP file
3. ✅ **Adjust confidence threshold** based on results
4. ✅ **Integrate into your workflow** (CLI, web app, or API)
5. ✅ **Collect feedback** and improve model with more data

---

## 📧 Need Help?

- Check training metrics in `results.csv`
- Review validation predictions in ZIP file
- Experiment with confidence thresholds
- Consider collecting more diverse training data

**Happy Detecting! 🏗️**


