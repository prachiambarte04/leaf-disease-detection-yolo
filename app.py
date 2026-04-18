import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 🔹 Page config
st.set_page_config(page_title="Leaf Disease Detection", layout="centered")

# 🔹 Load model
model = YOLO(r"D:\CNN_Object_Detection\models\leaf_model\weights\best.pt")

# 🔹 Title
st.title("🌿 Leaf Disease Detection")
st.write("Upload an image to detect plant diseases using YOLO")

# 🔹 Confidence slider (nice professional touch)
confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# 🔹 Upload image
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

# 🔹 If image uploaded
if uploaded_file is not None:
    
    image = Image.open(uploaded_file)

    # Show original
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    # 🔹 Prediction
    results = model.predict(img_array, conf=confidence)

    result_img = results[0].plot()

    # Show result
    st.image(result_img, caption="🎯 Detection Result", use_column_width=True)

    # 🔹 Details
    st.subheader("🔍 Detection Details")

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"✅ {model.names[cls]} → Confidence: {conf:.2f}")
    else:
        st.warning("⚠️ No object detected")

    st.success("✅ Detection Completed!")

else:
    st.info("👆 Please upload an image to start detection")