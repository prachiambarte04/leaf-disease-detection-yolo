import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Palm Leaf Detection",
    layout="wide"
)

# ---------------- BACKGROUND ----------------
def set_bg():

    with open(r"C:\Users\ADMIN\OneDrive\Music\Desktop\img.avif", "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .main .block-container {{
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(12px);
            border-radius: 25px;
            padding: 2rem;
        }}

        .hero-title {{
            text-align:center;
            color:white;
            font-size:58px;
            font-weight:800;
            margin-top:20px;
            text-shadow:2px 2px 10px rgba(0,0,0,0.8);
        }}

        .hero-subtitle {{
            text-align:center;
            color:#f1f1f1;
            font-size:24px;
            margin-bottom:25px;
            text-shadow:1px 1px 6px rgba(0,0,0,0.8);
        }}

        h1,h2,h3,p,label,div {{
            color:white !important;
        }}

        .glass-box {{
            background: rgba(255,255,255,0.12);
            padding:20px;
            border-radius:20px;
            backdrop-filter: blur(8px);
            text-align:center;
            min-height:220px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# ---------------- MODEL ----------------
model = YOLO(
    model = YOLO("weights/best.pt")
)

# ---------------- HERO TITLE ----------------
st.markdown(
    """
    <div class="hero-title">
    🌿 AI-Based Palm Leaf Nutrient Deficiency Detection System
    </div>

    <div class="hero-subtitle">
    AI Powered Crop Health Analysis using YOLOv8
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- LANGUAGE ----------------
language = st.selectbox(
    "🌐 Select Language",
    ["English", "Hindi"]
)

# ---------------- TEXT ----------------
text = {
    "English": {
        "detect":"🔍 Detect Now",
        "original":"📷 Original Image",
        "result":"🎯 Detection Result",
        "details":"🔍 Detection Details",
        "symptoms":"🩺 Symptoms",
        "treatment":"💊 Treatment",
        "wait":"Detecting Disease...",
        "download":"⬇ Download Result"
    },

    "Hindi": {
        "detect":"🔍 पहचान करें",
        "original":"📷 मूल फोटो",
        "result":"🎯 परिणाम",
        "details":"🔍 रोग विवरण",
        "symptoms":"🩺 लक्षण",
        "treatment":"💊 उपचार",
        "wait":"रोग पहचान हो रही है...",
        "download":"⬇ परिणाम डाउनलोड करें"
    }
}

# ---------------- DICTIONARIES ----------------

symptoms_dict = {

    "Healthy": {
        "English": """
• Uniform green color  
• No spots or damage
""",
        "Hindi": """
• समान हरा रंग  
• कोई धब्बे या नुकसान नहीं
"""
    },

    "Nitrogen": {
        "English": """
• Pale green or yellow leaves  
• Especially older leaves affected
""",
        "Hindi": """
• हल्के हरे या पीले पत्ते  
• विशेषकर पुराने पत्ते प्रभावित
"""
    },

    "Mg": {
        "English": """
• Yellowing between veins  
• Veins remain green
""",
        "Hindi": """
• नसों के बीच पीलापन  
• नसें हरी रहती हैं
"""
    },

    "Kalium": {
        "English": """
• Yellow or brown burnt edges  
• Dry leaf margins
""",
        "Hindi": """
• पीले या भूरे जले हुए किनारे  
• सूखे किनारे
"""
    },

    "Boron": {
        "English": """
• Deformed or twisted leaves  
• Irregular leaf growth
""",
        "Hindi": """
• विकृत या मुड़ी हुई पत्तियाँ  
• असामान्य वृद्धि
"""
    }
}

treatment_dict = {

    "Healthy": {
        "English": """
• No treatment required  
• Maintain proper care
""",
        "Hindi": """
• उपचार आवश्यक नहीं  
• सामान्य देखभाल रखें
"""
    },

    "Nitrogen": {
        "English": """
• Apply nitrogen fertilizer  
• Use compost or urea
""",
        "Hindi": """
• नाइट्रोजन उर्वरक दें  
• कम्पोस्ट या यूरिया उपयोग करें
"""
    },

    "Mg": {
        "English": """
• Apply magnesium sulfate  
• Improve soil nutrition
""",
        "Hindi": """
• मैग्नीशियम सल्फेट दें  
• मिट्टी की गुणवत्ता सुधारें
"""
    },

    "kalium": {
        "English": """
• Apply potash fertilizer  
• Maintain balanced irrigation
""",
        "Hindi": """
• पोटाश उर्वरक उपयोग करें  
• संतुलित सिंचाई रखें
"""
    },

    "Boron": {
        "English": """
• Apply boron fertilizer carefully  
• Avoid overuse
""",
        "Hindi": """
• बोरॉन उर्वरक सावधानी से दें  
• अधिक उपयोग न करें
"""
    }
}

# ---------------- IMAGE SOURCE ----------------

st.markdown(
    """
    <h2 style='text-align:center; color:white; margin-top:20px;'>
    📤 Choose Image Source
    </h2>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
image = None

box_style = """
background: rgba(255,255,255,0.12);
backdrop-filter: blur(10px);
border-radius: 25px;
padding: 15px;
height: 50px;
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
"""

# ---------------- UPLOAD BOX ----------------
with col1:

    st.markdown(
        "<h4 style='text-align:center;color:white;'>📤 Upload the Image</h4>",
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="{box_style}">',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        key="upload_box"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)

# ---------------- CAMERA BOX ----------------
with col2:

    st.markdown(
        "<h4 style='text-align:center;color:white;'>📷 Capture the Image</h4>",
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="{box_style}">',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.toggle(
        "📷 Camera ON / OFF",
        key="camera_on"
    )

    if st.session_state.get("camera_on", False):

        camera_image = st.camera_input(
            "",
            key="camera_box"
        )

        if camera_image:
            image = Image.open(camera_image)

    else:

        st.markdown(
            """
            <div style='
                height:220px;
                display:flex;
                justify-content:center;
                align-items:center;
                color:white;
                font-size:18px;
            '>
            📷 Camera Preview Disabled
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- DETECT BUTTON ----------------
detect = False

if image is not None:

    center = st.columns([1,1,1])

    with center[1]:
        detect = st.button(
            text[language]["detect"],
            use_container_width=True
        )

# ---------------- DETECTION ----------------

if detect:

    img_array = np.array(image)

    with st.spinner(text[language]["wait"]):

        results = model.predict(
            img_array,
            conf=0.25
        )

    result_img = results[0].plot()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Original Image
    with col1:
        st.image(
            image,
            caption=text[language]["original"],
            use_container_width=True
        )

    # Detection Image
    with col2:
        st.image(
            result_img,
            caption=text[language]["result"],
            use_container_width=True
        )

    st.subheader(text[language]["details"])

    if len(results[0].boxes) > 0:

        for box in results[0].boxes:

            cls = int(box.cls[0])
            conf_score = float(box.conf[0])

            # Get detected class
            disease = str(model.names[cls]).strip().lower()

            # Detection Result
            st.success(
                f"{disease} | Confidence: {conf_score:.2f}"
            )

            # ---------------- SYMPTOMS ----------------

            st.subheader(text[language]["symptoms"])

            matched_key = None

            for key in symptoms_dict.keys():
                if key.lower().replace(" ", "_") == disease.lower().replace(" ", "_"):
                    matched_key = key
                    break

            if matched_key:
                st.info(
                    symptoms_dict[matched_key][language]
                )
            else:
                st.error(
                    f"No symptom match for: {disease}"
                )

            # ---------------- TREATMENT ----------------
            
            st.subheader(text[language]["treatment"])

            matched_treatment = None

            for key in treatment_dict.keys():
                if key.lower().replace(" ", "_") == disease.lower().replace(" ", "_"):
                    matched_treatment = key
                    break

            if matched_treatment:
                st.success(
                    treatment_dict[matched_treatment][language]
                )
            else:
                st.error(
                    f"No treatment match for: {disease}"
                )
        
    # Download
    result_pil = Image.fromarray(result_img)

    buf = BytesIO()
    result_pil.save(buf, format="PNG")

    st.download_button(
        text[language]["download"],
        data=buf.getvalue(),
        file_name="leaf_detection.png",
        mime="image/png"
    )
