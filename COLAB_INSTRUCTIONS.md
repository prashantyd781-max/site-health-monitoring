# Google Colab Training Instructions

## 📚 Using the YOLOv8 Segmentation Training Notebook

### 1. Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `yolo_segmentation_training.ipynb`

### 2. Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **GPU** (T4 or better) as Hardware accelerator
3. Click **Save**

### 3. Get Roboflow API Key ⚠️ CRITICAL STEP

**This is the most important step! The notebook will not work without it.**

1. Go to [Roboflow Settings](https://app.roboflow.com/settings/api)
2. Log in or create a **FREE** account (no credit card required)
3. Copy your **Private API Key** (it looks like: `abc123XYZ456def789`)
4. In **Cell 6** (Dataset Download section), find this line:
   ```python
   ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"  # ← CHANGE THIS!
   ```
5. Replace `YOUR_API_KEY_HERE` with your actual key:
   ```python
   ROBOFLOW_API_KEY = "abc123XYZ456def789"  # Your actual key
   ```
6. **DO NOT** put quotes inside quotes or add extra spaces

**Example of CORRECT:**
```python
ROBOFLOW_API_KEY = "abcDEF123xyz789"
```

**Example of WRONG:**
```python
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"  # ❌ Not replaced!
ROBOFLOW_API_KEY = ""abcDEF123xyz789""  # ❌ Extra quotes!
ROBOFLOW_API_KEY = " abcDEF123xyz789 "  # ❌ Extra spaces!
```

### 4. Run the Notebook

**Option 1: Run All Cells**
- Click **Runtime → Run all**
- The training will start automatically

**Option 2: Run Step by Step**
- Click on each cell and press `Shift + Enter` to run
- This allows you to see results at each step

### 5. Monitor Training

The training will:
- Take approximately 2-4 hours (depending on GPU)
- Show real-time progress with loss curves
- Display validation predictions after each epoch
- Save checkpoints automatically

### 6. Download Results

After training completes, the notebook will:
- Automatically download `best.pt` (best model)
- Download `last.pt` (last checkpoint)
- Create and download a ZIP file with all results

## 📊 What You'll Get

### Model Files
- `best.pt` - Best performing model (use this for deployment)
- `last.pt` - Final epoch checkpoint

### Training Results
- `results.csv` - Metrics for each epoch
- `results.png` - Training curves visualization
- `confusion_matrix.png` - Model confusion matrix
- `val_batch*_pred.jpg` - Validation predictions

## 🎯 Training Configuration

Default settings in the notebook:
```python
epochs: 100          # Training iterations
batch: 16            # Batch size (adjust if GPU runs out of memory)
imgsz: 640          # Image size
patience: 20        # Early stopping patience
optimizer: Adam     # Optimization algorithm
lr0: 0.001         # Initial learning rate
```

### Adjusting Batch Size
If you get **CUDA out of memory** error:
1. Go to Cell 8 (Configure Training Parameters)
2. Change `'batch': 16` to a smaller value (e.g., `8` or `4`)
3. Rerun from that cell

## 💡 Tips

### For Faster Training
- Use **GPU Premium** runtime (A100 or V100)
- Reduce image size: `'imgsz': 512` or `'imgsz': 416`
- Use a smaller model: `'model': 'yolov8n-seg.pt'` (nano)

### For Better Accuracy
- Increase epochs: `'epochs': 150`
- Use larger model: `'model': 'yolov8m-seg.pt'` (medium)
- Keep image size at 640

### For Your Own Dataset
Replace Cell 6 with your own Roboflow project:
```python
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(VERSION_NUMBER).download("yolov8")
```

## 🔧 Troubleshooting

### NumPy Compatibility Error
**Error:** `numpy.core.multiarray failed to import` or `AttributeError: _ARRAY_API not found`

**What happened:** Google Colab updated to NumPy 2.x, but matplotlib and other packages need NumPy 1.x

**Solution:**
1. The notebook (Cell 4) now automatically downgrades NumPy to 1.x
2. If you still get this error:
   - Click `Runtime → Restart runtime`
   - Run all cells again from the beginning
   - Cell 4 should show "NumPy: 1.x.x" (not 2.x.x)

### Runtime Disconnects
Google Colab may disconnect after 12 hours or if idle. To prevent:
- Keep the browser tab active
- Use Colab Pro for longer sessions
- The training will resume from last checkpoint if interrupted

### GPU Not Available
If no GPU is detected:
- Check Runtime → Change runtime type
- Free tier has limited GPU access
- Try again later or upgrade to Colab Pro

### Dataset Download Fails
- Verify your Roboflow API key is correct
- Check your internet connection
- Ensure the dataset exists in your Roboflow account

## 📞 Using the Trained Model

After downloading `best.pt`, you can use it locally:

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run inference
results = model('your_image.jpg')

# Display
results[0].show()

# Save result
results[0].save('output.jpg')
```

## 🚀 Next Steps

1. Test the model on your own images (Cell 8)
2. Export to different formats (Cell 10)
3. Deploy in your application
4. Fine-tune with more data if needed

---

**Happy Training! 🎯**

For questions or issues, check the [Ultralytics Documentation](https://docs.ultralytics.com/)

