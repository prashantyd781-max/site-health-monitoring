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

st.sidebar.subheader("📁 Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"]
)


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
    if yolo_model is None:
        st.error("❌ YOLOv11 detection model not loaded!")
        return image_np
    
    results = yolo_model(image_np)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            width_px = x2 - x1
            height_px = y2 - y1
            width_cm = width_px * px_to_cm_ratio
            height_cm = height_px * px_to_cm_ratio

            class_id = int(box.cls[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()
            label = yolo_model.names[class_id]

            # Draw the bounding box and label on the image
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dimension_text = (
                f"         {width_cm:.2f}cm x {height_cm:.2f}cm ({confidence:.2f})"
            )
            cv2.putText(
                image_np,
                dimension_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    annotated_image = results[0].plot()
    return annotated_image


def segment_image(image_np):

    if segmentation_model is None:
        st.error("❌ Segmentation model not loaded!")
        return image_np
    
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)

    segmented_image = results[0].plot()

    return segmented_image


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing the image..."):
        time.sleep(2)

        processed_image = detect_with_yolo(image_np)
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        segmented_image = segment_image(image_np)

        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        equalized_image = preprocess_image_for_depth_estimation(image_np)
        depth_heatmap = create_depth_estimation_heatmap(equalized_image)
        edges = apply_canny_edge_detection(image_np)

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            processed_image,
            caption="Crack Detection Results",
            use_column_width=True,
        )
        st.image(
            depth_heatmap, caption="Depth Estimation Heatmap", use_column_width=True
        )
    with col2:
        st.image(segmented_image, caption="Segmentation Result", use_column_width=True)
        st.image(edges, caption="Canny Edge Detection", use_column_width=True)


st.markdown(
    '<div class="footer">© 2024 Heritage Health Monitoring <i class="fas fa-globe"></i></div>',
    unsafe_allow_html=True,
)
