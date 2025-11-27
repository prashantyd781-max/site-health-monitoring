# 🔥 Kernel Dying / Crashing - QUICK FIX

## The Problem

Your Kaggle kernel is dying/restarting during training with error:
```
Kernel is dying
```
or the cell just stops and kernel restarts.

---

## 🎯 THE CAUSE

**Out of Memory (OOM)** - Your GPU ran out of memory!

This happens when:
- Batch size is too large for available GPU memory
- Image size is too large
- Model is too complex for the GPU

**Most common:** Batch size = 16 is too much for Kaggle's T4 GPU with segmentation models

---

## ✅ INSTANT FIX (3 Steps)

### Step 1: Reduce Batch Size

In the **training configuration cell**, change:

**Before (causes crash):**
```python
'batch': 16,  # ❌ Too large!
```

**After (should work):**
```python
'batch': 8,   # ✅ Safer
```

**If still crashing:**
```python
'batch': 4,   # ✅ Very safe
```

**Last resort:**
```python
'batch': 2,   # ✅ Will definitely work (but slower)
```

### Step 2: Clear GPU Memory

Add this cell BEFORE training:
```python
import torch
torch.cuda.empty_cache()
print("✅ GPU cache cleared")
```

### Step 3: Re-run Training

- Run the training cell again
- It should now complete without crashing

---

## 🔍 How to Check GPU Memory

**Before training**, run this:

```python
import torch

if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total GPU Memory: {gpu_memory:.2f} GB")
    
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"Currently Used: {allocated:.2f} GB")
    print(f"Available: {gpu_memory - allocated:.2f} GB")
    
    # Recommendations
    if gpu_memory < 16:
        print("\n⚠️  Recommended batch size: 4-8")
    else:
        print("\n✅ Can use batch size: 8-16")
```

---

## 📊 Batch Size Guide

| GPU | GPU Memory | Recommended Batch Size | Training Time |
|-----|------------|------------------------|---------------|
| T4 | ~15 GB | **4-8** | 3-5 hours |
| P100 | ~16 GB | **8** | 2-4 hours |
| V100 | ~32 GB | 16-32 | 1-2 hours |

For **Kaggle T4 x2**: Use batch size **8** (already updated in notebook)

---

## 🎯 Your Specific Fix

The notebook has been updated with:

1. ✅ **Batch size reduced to 8** (from 16)
2. ✅ **GPU memory check cell** added (Cell 9)
3. ✅ **Clear instructions** in troubleshooting

**What to do NOW:**

### Option 1: Use Updated Configuration (Recommended)
Just run all cells - batch size is already set to 8!

### Option 2: If Still Crashing
Add this BEFORE the training cell:
```python
CONFIG['batch'] = 4  # Override to batch size 4
```

Then run training cell.

---

## 🔄 Recovery Steps

If kernel already died:

### Step 1: Don't Panic!
Your dataset is still downloaded (unless session ended)

### Step 2: Check What's Saved
```python
!ls -lh /kaggle/working/runs/segment/*/weights/*.pt 2>/dev/null
```

If you see `last.pt` - you can resume training!

### Step 3: Resume Training (if possible)
```python
# Change batch size first!
CONFIG['batch'] = 4

# Resume from last checkpoint
CONFIG['resume'] = True

# Re-run training
results = model.train(**CONFIG)
```

### Step 4: Start Fresh (if needed)
1. Reduce batch size to 4
2. Re-run training cell
3. Should complete successfully

---

## 💡 Other Solutions

### Reduce Image Size
```python
'imgsz': 512,  # Instead of 640
```

### Reduce Epochs (for testing)
```python
'epochs': 50,  # Instead of 100
```

### Use Smaller Model
```python
'model': 'yolov8n-seg.pt',  # Already using smallest!
```

### Mixed Precision (already enabled)
```python
'amp': True,  # Automatic mixed precision
```

---

## 📈 Performance Impact

### Batch Size 16 vs 8 vs 4:

| Batch Size | GPU Usage | Speed | Training Time | Stability |
|------------|-----------|-------|---------------|-----------|
| 16 | ~14 GB | Fast | 2-3h | ❌ Crashes |
| **8** | ~8 GB | Medium | **3-4h** | **✅ Good** |
| 4 | ~5 GB | Slower | 4-6h | ✅ Very Stable |
| 2 | ~3 GB | Slowest | 6-8h | ✅ Ultra Stable |

**Recommended for Kaggle: Batch Size 8** (already set!)

---

## ✅ Verification

After making changes, you should see:

### During Training:
```
Epoch   GPU_mem   box_loss   seg_loss   cls_loss
1/100     7.2G      1.234      0.567      0.891  ✅ Good!
```

**GPU_mem should be < 10 GB on T4**

### If Crashing:
```
Epoch   GPU_mem   box_loss   seg_loss   cls_loss
1/100    14.8G      1.234      0.567      0.891  ❌ About to crash!
[Kernel dies]
```

**GPU_mem > 14 GB = Will crash soon!**

---

## 🚨 Emergency Workaround

If nothing works, use this minimal config:

```python
CONFIG = {
    'model': 'yolov8n-seg.pt',
    'data': f'{dataset_path}/data.yaml',
    'epochs': 50,        # Reduced
    'imgsz': 416,        # Much smaller
    'batch': 4,          # Very safe
    'device': device,
    'project': '/kaggle/working/runs/segment',
    'name': 'spalling_rebar_emergency',
    'patience': 20,
    'save': True,
    'plots': True,
    'verbose': True,
}
```

This will definitely work (but with lower accuracy).

---

## 📊 Expected Behavior

### ✅ Normal Training:
```
Epoch   GPU_mem   box_loss   seg_loss   cls_loss   Instances   Size
1/100     7.2G      1.234      0.567      0.891      45         640
2/100     7.2G      1.123      0.534      0.867      43         640
3/100     7.2G      1.045      0.512      0.834      47         640
...continues normally...
```

### ❌ About to Crash:
```
Epoch   GPU_mem   box_loss   seg_loss   cls_loss   Instances   Size
1/100    14.8G      1.234      0.567      0.891      45         640
[Kernel restarts - OOM!]
```

---

## 🎉 Summary

**Quick Fix Checklist:**
- [ ] Reduce batch size to 8 (already done in updated notebook!)
- [ ] Clear GPU cache before training
- [ ] Check GPU memory (Cell 9)
- [ ] If still crashes → batch size 4
- [ ] If STILL crashes → batch size 2 + imgsz 512

**The updated notebook already has batch=8, so just re-run it and it should work!**

---

## 🔗 Related Issues

- **NumPy error**: See `NUMPY_FIX.md`
- **API key error**: See `API_KEY_GUIDE.md`
- **General help**: See `KAGGLE_INSTRUCTIONS.md`

---

**Your kernel should not crash anymore with batch size 8!** 🚀

If you're still having issues, drop to batch size 4 and it will definitely work!


