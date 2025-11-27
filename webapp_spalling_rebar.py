#!/usr/bin/env python3
"""
Spalling & Exposed Rebar Detection Web App
Run: streamlit run webapp_spalling_rebar.py
"""

try:
    import streamlit as st
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
    import os
    
    # Page config
    st.set_page_config(
        page_title="Spalling & Rebar Detector",
        page_icon="🏗️",
        layout="wide"
    )
    
    # Title
    st.title("🏗️ Spalling & Exposed Rebar Detection")
    st.markdown("Upload an image to detect structural defects")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Use trained model (no option to change)
    model_path = "best.pt"
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.info("Detects:\n- 🧱 Spalling (concrete deterioration)\n- 🔩 Exposed Rebar")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.info("💡 Make sure 'best.pt' is in the same directory as this script")
        st.stop()
    
    # Load model
    @st.cache_resource
    def load_model(path):
        return YOLO(path)
    
    try:
        model = load_model(model_path)
        st.sidebar.success("✅ Trained model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of concrete structure"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Image info
            st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
        
        # Run detection
        with st.spinner("🔍 Analyzing image..."):
            results = model(image, conf=confidence, verbose=False)
            result = results[0]
        
        # Display annotated image
        with col2:
            st.subheader("🎯 Detection Results")
            
            # Convert result to image
            annotated_img = result.plot()
            annotated_img = Image.fromarray(annotated_img[..., ::-1])  # BGR to RGB
            st.image(annotated_img, use_container_width=True)
        
        # Show detection details
        st.markdown("---")
        st.subheader("📊 Detection Details")
        
        boxes = result.boxes
        
        if len(boxes) > 0:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            # Count by class
            class_counts = {}
            for box in boxes:
                cls_name = model.names[int(box.cls[0])]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            with col1:
                st.metric("Total Detections", len(boxes))
            
            with col2:
                spalling_count = class_counts.get('Spalling', 0)
                st.metric("🧱 Spalling", spalling_count)
            
            with col3:
                rebar_count = class_counts.get('Exposed-Rebar', 0)
                st.metric("🔩 Exposed Rebar", rebar_count)
            
            # Detailed table
            st.markdown("### 📋 Detailed Detections")
            
            detections_data = []
            for i, box in enumerate(boxes, 1):
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections_data.append({
                    "#": i,
                    "Class": cls_name,
                    "Confidence": f"{conf:.2%}",
                    "Location": f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
                })
            
            st.table(detections_data)
            
            # Download button
            import io
            buf = io.BytesIO()
            annotated_img.save(buf, format='JPEG')
            
            st.download_button(
                label="💾 Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"detected_{uploaded_file.name}",
                mime="image/jpeg"
            )
            
        else:
            st.info(f"ℹ️ No defects detected above {confidence:.0%} confidence threshold")
            st.markdown("💡 Try lowering the confidence threshold in the sidebar")
    
    else:
        # Instructions
        st.info("👆 Upload an image to get started")
        
        st.markdown("### 🎯 What This App Does")
        st.markdown("""
        This app uses your trained YOLOv8 model to detect:
        - **Spalling**: Concrete deterioration and cracking
        - **Exposed Rebar**: Visible reinforcement bars
        
        ### 📝 How to Use
        1. Upload an image using the file uploader above
        2. Adjust confidence threshold in the sidebar (if needed)
        3. View detection results
        4. Download the annotated image
        
        ### 💡 Tips
        - Use **0.25** confidence for balanced results
        - Lower to **0.15** to catch more defects
        - Raise to **0.40** for high-confidence only
        """)

except ImportError as e:
    print("\n❌ Error: Streamlit not installed!")
    print("\n📦 Install required packages:")
    print("   pip install streamlit ultralytics pillow")
    print("\n🚀 Then run:")
    print("   streamlit run webapp_spalling_rebar.py")
    print()

