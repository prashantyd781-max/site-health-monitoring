# 🚀 Kaggle Training Instructions

## Using the YOLOv8 Segmentation Training Notebook on Kaggle

---

## 📋 Quick Setup (3 minutes)

### Step 1: Upload Notebook to Kaggle
1. Go to https://www.kaggle.com/
2. Log in to your account
3. Click **"Code"** → **"New Notebook"**
4. Or click **"File"** → **"Import Notebook"**
5. Upload `yolo_segmentation_training_kaggle.ipynb`

### Step 2: Enable GPU & Internet
1. Click **"Settings"** on the right panel
2. **Accelerator**: Select **"GPU T4 x2"** (or P100 if available)
3. **Internet**: Toggle **ON** (required for dataset download)
4. **Persistence**: Toggle **ON** (saves your work)

### Step 3: Get Roboflow API Key
1. Go to https://app.roboflow.com/settings/api
2. Log in or create FREE account (no credit card)
3. Copy your **Private API Key**
4. Paste it in **Cell 6** of the notebook

### Step 4: Run Training
1. Click **"Run All"** (or Shift+Enter for each cell)
2. Wait 2-4 hours for training to complete
3. Download results from the **"Output"** tab

---

## ⚡ Key Differences from Google Colab

### File Paths
- **Kaggle**: `/kaggle/working/` (saved as notebook output)
- **Colab**: `/content/` (temporary, deleted after session)

### GPU Access
- **Kaggle**: 30 hours/week free GPU (T4 x2)
- **Colab**: ~12 hours/day free GPU (T4)

### Internet
- **Kaggle**: Must be manually enabled (Settings → Internet → ON)
- **Colab**: Always available

### File Download
- **Kaggle**: Files in `/kaggle/working/` → Output tab
- **Colab**: Use `files.download()` function

### Session Duration
- **Kaggle**: 9 hours max per session (upgradable)
- **Colab**: 12 hours max (24 hours with Pro)

---

## 🔧 Important Settings for Kaggle

### Enable GPU (Required!)
```
Settings → Accelerator → GPU T4 x2
```
Without GPU, training will take days instead of hours!

### Enable Internet (Required!)
```
Settings → Internet → ON
```
Required to download dataset from Roboflow and install packages.

### Enable Persistence (Recommended)
```
Settings → Persistence → ON
```
Saves intermediate checkpoints if session disconnects.

---

## 📂 Where Files Are Saved

### Kaggle Working Directory
All files in `/kaggle/working/` are automatically saved as notebook output:
```
/kaggle/working/
├── best.pt                    ← Your trained model
├── last.pt                    ← Last checkpoint
├── best.onnx                  ← ONNX export (optional)
└── runs/
    └── segment/
        └── spalling_rebar_training/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── results.csv
            ├── results.png
            ├── confusion_matrix.png
            └── [other files]
```

### How to Download
1. Click the **"Output"** tab at the top of the page
2. Find your files listed there
3. Click the **download icon** next to each file
4. Or click **"Download All"** to get everything as a ZIP

---

## ✅ What to Expect

### After Cell 4 (Installation):
```
✅ All dependencies installed successfully!
✅ NumPy version fixed for compatibility

📊 Installed versions:
   NumPy: 1.26.4              ← Should be 1.x.x
   Ultralytics: ✓ Installed
   Roboflow: ✓ Installed
```

### After Cell 6 (Dataset Download):
```
✅ Dataset downloaded successfully!
📂 Location: /kaggle/working/Spalling-and-exposed-rebar-1

📊 Dataset Structure:
   train : 2997 images
   valid : 120 images
   test  : 108 images
```

### During Training (Cell 11):
```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
1/100     4.2G      1.234      0.567      0.891      45         640
...
```

### After Training Complete:
```
🎉 TRAINING COMPLETED! 🎉
⏱️  Total training time: 2.5 hours
📂 Model saved to: /kaggle/working/runs/segment/spalling_rebar_training/weights/best.pt
```

---

## 🆘 Troubleshooting

### Issue #1: NumPy Compatibility Error
**Error:** `numpy.core.multiarray failed to import`

**Solution:**
- Already fixed in Cell 4!
- Cell 4 automatically downgrades NumPy to 1.x
- If error persists: **Session → Restart** and run all cells again

---

### Issue #2: API Key Error  
**Error:** `RoboflowError: This API key does not exist`

**Solution:**
- You need to replace `"YOUR_API_KEY_HERE"` in Cell 6
- Get your key from: https://app.roboflow.com/settings/api
- Paste it between the quotes (no extra spaces)

---

### Issue #3: Internet Not Available
**Error:** `URLError` or `Connection refused` or `No internet connection`

**Solution:**
- Go to: Settings → Internet → Turn **ON**
- Re-run Cell 6 (dataset download)
- Kaggle requires manual internet activation for security

---

### Issue #4: GPU Not Available
**Error:** Training is very slow / "No GPU detected"

**Solution:**
- Settings → Accelerator → Select **GPU T4 x2**
- If GPU quota exceeded: Wait for weekly reset
- Upgrade to Kaggle Pro for more GPU hours

---

### Issue #5: Session Timeout / Disconnect
**Error:** Training interrupted after 9 hours

**Solution:**
- Kaggle free tier: 9-hour max session
- Enable Persistence (Settings → Persistence → ON)
- Training will resume from last checkpoint
- Consider splitting into smaller epoch batches

---

### Issue #6: Out of Memory
**Error:** `CUDA out of memory` or `RuntimeError: out of memory`

**Solution:**
- In Cell 9, reduce batch size:
  ```python
  'batch': 8,  # Changed from 16
  ```
- Or reduce image size:
  ```python
  'imgsz': 512,  # Changed from 640
  ```
- Re-run from Cell 9 onwards

---

## 💡 Tips for Kaggle

### Save GPU Quota
- Kaggle gives **30 hours/week** of free GPU
- Training takes ~2-4 hours
- Close notebook when not using to save quota
- Check quota: Profile → Settings → Quotas

### Faster Training
- Use **GPU P100** instead of T4 (faster but uses more quota)
- Reduce epochs: `'epochs': 50` instead of 100
- Use smaller model: `yolov8n-seg.pt` (already the smallest)

### Better Accuracy
- Increase epochs: `'epochs': 150`
- Use larger model: `yolov8m-seg.pt` (medium)
- Increase image size: `'imgsz': 800`
- But these will take longer!

### Use Your Own Dataset
Replace Cell 6 with your Roboflow project:
```python
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(VERSION).download("yolov8")
```

---

## 📊 Performance Expectations

### Training Time:
- **T4 x2 GPU**: 2-4 hours
- **P100 GPU**: 1.5-3 hours
- **CPU only**: 24-48 hours (not recommended!)

### Model Performance:
- **Box mAP50**: 75-85%
- **Mask mAP50**: 70-80%
- **Model Size**: ~6MB
- **Inference Speed**: ~50ms per image (on GPU)

---

## 🎯 After Training

### Download Your Model
1. Go to **"Output"** tab
2. Download `best.pt` (main model)
3. Download `results.csv` (metrics)
4. Optionally download entire output as ZIP

### Use Your Model
```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run on an image
results = model('your_image.jpg')

# Display
results[0].show()

# Save
results[0].save('output.jpg')
```

### Share Your Notebook
1. Click **"Share"** button
2. Set visibility to **Public** (optional)
3. Share the link with others
4. They can fork and run it themselves!

---

## 📚 Additional Resources

### Kaggle-Specific:
- **Kaggle GPU Quotas**: https://www.kaggle.com/docs/notebooks#gpu
- **Kaggle Output Files**: https://www.kaggle.com/docs/notebooks#output
- **Kaggle Community**: https://www.kaggle.com/discussions

### YOLO & Training:
- **Ultralytics Docs**: https://docs.ultralytics.com/
- **Roboflow Docs**: https://docs.roboflow.com/
- **YOLOv8 Tutorial**: https://docs.ultralytics.com/tasks/segment/

---

## ✅ Quick Checklist

Before running:
- [ ] Notebook uploaded to Kaggle
- [ ] GPU enabled (Settings → Accelerator → GPU T4 x2)
- [ ] Internet enabled (Settings → Internet → ON)
- [ ] Roboflow account created (free)
- [ ] API key obtained and pasted in Cell 6
- [ ] Ready to click "Run All"!

During training:
- [ ] Check GPU usage in system monitor
- [ ] Watch training progress (loss decreasing)
- [ ] Keep browser tab active (or enable persistence)

After training:
- [ ] Check Output tab for files
- [ ] Download best.pt model
- [ ] Test on your own images
- [ ] Share your results!

---

## 🎉 Summary

**Setup**: 3 minutes  
**Training**: 2-4 hours  
**Download**: 1 minute  
**Total**: ~3 hours (mostly unattended)

### What You Get:
✅ Trained YOLOv8 segmentation model  
✅ Performance metrics and visualizations  
✅ Ready-to-use `best.pt` file  
✅ All training logs and checkpoints  

### Files Saved Automatically:
- `/kaggle/working/best.pt` → Output tab
- `/kaggle/working/runs/` → Output tab
- Everything accessible via Output tab!

---

**Happy Training on Kaggle! 🚀**

*The notebook is optimized for Kaggle and handles all the platform-specific requirements automatically!*


