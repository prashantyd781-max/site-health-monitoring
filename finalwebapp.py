import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import time
from ultralytics import YOLO
from pathlib import Path

# IP Webcam stream URL (update here if needed)
IP_WEBCAM_URL = "http://100.96.4.31:8080/video"

st.set_page_config(page_title="Heritage Sites Health Monitoring", layout="wide")

BASE_DIR = Path(__file__).resolve().parent  # demo/

# Use the combined model (detects ALL defects)
available_models = {
    "YOLOv8 (Working)": {
        "path": BASE_DIR / "combined_all_defects_best.pt",  # Combined model
        "backup": BASE_DIR / "yolov8_crack_detection_WORKING.pt",  # Fallback
        "version": "YOLOv8", 
        "description": "Combined Model: Cracks + Spalling + Exposed Rebar"
    }
}

segmentation_model_path = BASE_DIR / "segmentation_model" / "weights" / "best.pt"


st.sidebar.header("Options")

# Automatically use YOLOv8 (no selection dropdown)
selected_model_name = "YOLOv8 (Working)"

# Function to load selected model
def load_selected_model(model_name):
    if model_name is None:
        return None, "❌ No model selected"
    
    model_info = available_models[model_name]
    model_path = model_info["path"]
    backup_path = model_info["backup"]
    
    # Try primary path first, then backup
    if model_path.exists():
        try:
            model = YOLO(str(model_path))
            status = f"✅ Loaded {model_info['version']} from: {model_path.name}"
            return model, status
        except Exception as e:
            if backup_path and backup_path.exists():
                try:
                    model = YOLO(str(backup_path))
                    status = f"✅ Loaded {model_info['version']} from backup: {backup_path.name}"
                    return model, status
                except Exception as e2:
                    return None, f"❌ Failed to load model: {str(e2)}"
            else:
                return None, f"❌ Failed to load model: {str(e)}"
    elif backup_path and backup_path.exists():
        try:
            model = YOLO(str(backup_path))
            status = f"✅ Loaded {model_info['version']} from backup: {backup_path.name}"
            return model, status
        except Exception as e:
            return None, f"❌ Failed to load backup model: {str(e)}"
    else:
        return None, f"❌ Model files not found: {model_path.name}"

# Load the selected model
yolo_model, model_status = load_selected_model(selected_model_name)

# Display model status
if yolo_model:
    st.sidebar.success("✅ Combined YOLOv8 model loaded")
    st.sidebar.info("🎯 Detects: Cracks, Spalling & Exposed Rebar")
else:
    st.sidebar.error(model_status)


# Load segmentation model  
st.sidebar.subheader("🎯 Segmentation Model")
if segmentation_model_path.exists():
    try:
        segmentation_model = YOLO(str(segmentation_model_path))
        st.sidebar.success(f"✅ Segmentation model loaded: {segmentation_model_path.name}")
    except Exception as e:
        segmentation_model = None
        st.sidebar.error(f"❌ Failed to load segmentation model: {str(e)}")
else:
    segmentation_model = None
    st.sidebar.warning("⚠️ Segmentation model not found")

st.sidebar.subheader("📁 Upload Images")
uploaded_files = st.sidebar.file_uploader(
    "Choose images (max 100)...", 
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# Limit to 100 images
if uploaded_files and len(uploaded_files) > 100:
    st.sidebar.warning("⚠️ Maximum 100 images allowed. Only first 100 will be processed.")
    uploaded_files = uploaded_files[:100]

if uploaded_files:
    st.sidebar.info(f"📊 {len(uploaded_files)} image(s) selected")


px_to_cm_ratio = 0.1


def get_ip_webcam_frame():
    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        st.error("Could not fetch frame from IP webcam.")
        return None

def preprocess_image_for_depth_estimation(image_np):
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    equalized_image = cv2.equalizeHist(blurred_image)
    return equalized_image


def create_depth_estimation_heatmap(equalized_image):
    _, shadow_mask = cv2.threshold(equalized_image, 60, 255, cv2.THRESH_BINARY_INV)
    shadow_region = cv2.bitwise_and(equalized_image, equalized_image, mask=shadow_mask)
    depth_estimation = 255 - shadow_region
    depth_estimation_normalized = cv2.normalize(
        depth_estimation, None, 0, 255, cv2.NORM_MINMAX
    )
    depth_heatmap_colored = cv2.applyColorMap(
        depth_estimation_normalized.astype(np.uint8), cv2.COLORMAP_JET
    )
    return depth_heatmap_colored


def apply_canny_edge_detection(image_np):
    edges = cv2.Canny(image_np, 100, 200)
    return edges


def detect_with_yolo(image_np):
    """
    Detect defects using YOLO model.
    Returns: (annotated_image, defect_counts dict)
    """
    if yolo_model is None:
        st.error("❌ YOLOv11 detection model not loaded!")
        return image_np, {}
    
    results = yolo_model(image_np)
    result = results[0]
    
    # Count detections directly from class tensor
    defect_counts = {}
    
    # Get number of detections using len()
    num_boxes = len(result.boxes) if result.boxes is not None else 0
    
    if num_boxes > 0:
        # Get all class IDs at once
        cls_tensor = result.boxes.cls
        for i in range(num_boxes):
            cls_id = int(cls_tensor[i].item())
            label = yolo_model.names[cls_id]
            defect_counts[label] = defect_counts.get(label, 0) + 1

    annotated_image = result.plot()
    return annotated_image, defect_counts


def segment_image(image_np):
    """
    Segment image and return both the segmented image and defect counts.
    Returns: (segmented_image, defect_counts dict)
    """
    if segmentation_model is None:
        st.error("❌ Segmentation model not loaded!")
        return image_np, {}
    
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)
    result = results[0]
    
    # Count detections from segmentation model
    defect_counts = {}
    num_boxes = len(result.boxes) if result.boxes is not None else 0
    
    if num_boxes > 0:
        cls_tensor = result.boxes.cls
        for i in range(num_boxes):
            cls_id = int(cls_tensor[i].item())
            label = segmentation_model.names[cls_id]
            defect_counts[label] = defect_counts.get(label, 0) + 1

    segmented_image = result.plot()

    return segmented_image, defect_counts


if uploaded_files:
    total_images = len(uploaded_files)
    
    # Initialize total defect counts
    total_defects = {}
    
    st.subheader(f"📸 Processing {total_images} Image(s)")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    defect_summary = st.empty()
    
    # Store all results
    all_results = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (idx + 1) / total_images
        progress_bar.progress(progress)
        status_text.markdown(f"**🔄 Processing:** {idx + 1} / {total_images} images | **Current:** {uploaded_file.name}")
        
        # Update running defect count
        total_defect_count = sum(total_defects.values())
        if total_defects:
            defect_str = " | ".join([f"{k}: {v}" for k, v in total_defects.items()])
            defect_summary.markdown(f"**🔍 Total Defects Found:** {total_defect_count} ({defect_str})")
        
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Detection (for visualization)
        processed_image, detection_defects = detect_with_yolo(image_np.copy())
        
        # Segmentation (this is where cracks are actually detected!)
        segmented_image, segmentation_defects = segment_image(image_np.copy())
        
        # Use segmentation defects as the primary source (it detects cracks better)
        image_defects = segmentation_defects
        
        # Aggregate defect counts
        for defect_type, count in image_defects.items():
            if defect_type not in total_defects:
                total_defects[defect_type] = 0
            total_defects[defect_type] += count
        
        # Depth and edge detection
        equalized_image = preprocess_image_for_depth_estimation(image_np.copy())
        depth_heatmap = create_depth_estimation_heatmap(equalized_image)
        edges = apply_canny_edge_detection(image_np.copy())
        
        all_results.append({
            "name": uploaded_file.name,
            "original": image,
            "processed": processed_image,
            "segmented": segmented_image,
            "depth": depth_heatmap,
            "edges": edges,
            "defects": image_defects
        })
    
    # Final status update
    status_text.markdown(f"**✅ Completed:** {total_images} / {total_images} images")
    
    # Final defect summary
    total_defect_count = sum(total_defects.values())
    st.markdown("---")
    st.subheader("📊 Analysis Summary")
    
    # Create summary metrics
    col_summary = st.columns(4)
    with col_summary[0]:
        st.metric("📷 Images Processed", total_images)
    with col_summary[1]:
        st.metric("🔍 Total Defects", total_defect_count)
    with col_summary[2]:
        # Count cracks (case-insensitive matching)
        crack_count = sum(count for defect_type, count in total_defects.items() 
                         if "crack" in defect_type.lower())
        st.metric("🔴 Cracks", crack_count)
    with col_summary[3]:
        other_defects = total_defect_count - crack_count
        st.metric("⚠️ Other Defects", other_defects)
    
    # Detailed breakdown
    if total_defects:
        st.markdown("**Defect Breakdown:**")
        defect_cols = st.columns(len(total_defects) if len(total_defects) <= 5 else 5)
        for i, (defect_type, count) in enumerate(total_defects.items()):
            with defect_cols[i % len(defect_cols)]:
                st.info(f"**{defect_type}:** {count}")
    
    st.markdown("---")
    
    # Display results for each image
    for i, result in enumerate(all_results):
        with st.expander(f"📷 Image {i+1}: {result['name']} ({sum(result['defects'].values())} defects)", expanded=(i == 0)):
            st.image(result['original'], caption="Original Image", use_container_width=True)
            
            # Show defects found in this image
            if result['defects']:
                defect_str = ", ".join([f"{k}: {v}" for k, v in result['defects'].items()])
                st.success(f"🔍 Defects found: {defect_str}")
            else:
                st.info("✅ No defects detected in this image")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    result['processed'],
                    caption="Crack Detection Results",
                    use_container_width=True,
                )
                st.image(
                    result['depth'], caption="Depth Estimation Heatmap", use_container_width=True
                )
            with col2:
                st.image(result['segmented'], caption="Segmentation Result", use_container_width=True)
                st.image(result['edges'], caption="Canny Edge Detection", use_container_width=True)


st.markdown(
    '<div class="footer">© 2024 Heritage Health Monitoring <i class="fas fa-globe"></i></div>',
    unsafe_allow_html=True,
)
