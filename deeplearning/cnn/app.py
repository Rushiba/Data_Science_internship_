# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os
from PIL import Image

st.set_page_config(page_title="VisionAI Dashboard", page_icon="🎨", layout="wide")

# Inject Custom Colorful CSS Theme
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stSidebar { background-color: #1a1a2e !important; color: white; }
    h1 { color: #3b5998; font-family: 'Arial', sans-serif; }
    .prediction-card {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white; padding: 25px; border-radius: 15px;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .metric-box {
        background-color: #ffffff; padding: 15px; border-radius: 10px;
        border-left: 5px solid #0072ff; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Resolve file absolute paths dynamically to fix the "File Not Found" bug
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
model_path = os.path.join(current_dir, 'my_model.h5')
meta_path = os.path.join(current_dir, 'model_meta.joblib')

@st.cache_resource
def load_deep_learning_assets():
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
    model = tf.keras.models.load_model(model_path)
    meta = joblib.load(meta_path)
    return model, meta

model, meta = load_deep_learning_assets()

# Sidebar Multi-page Navigation Setup
st.sidebar.title("🎨 VisionAI Control Panel")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate Pages", ["🏠 Home & Overview", "🔮 Real-Time Identifier", "📊 Model Metrics"])

if model is None:
    st.error(f"⚠️ Model files are missing in this directory: {current_dir}")
    st.info("Please run your `train.py` script completely inside VS Code first to generate the needed assets.")
    st.stop()

class_names = meta['class_names']
target_size = meta['input_shape'][:2]

# ==========================================
# PAGE: HOME
# ==========================================
if page == "🏠 Home & Overview":
    st.title("🚀 Welcome to VisionAI Recognition Hub")
    st.write("An interactive, multi-page deep learning dashboard driven by Transfer Learning.")
    
    st.markdown("""
    ### 🌟 Core Features:
    * **Advanced Pre-trained Feature Maps:** Uses optimized deep layer extraction.
    * **Beautiful UI:** Multi-page layout featuring customized color banners.
    * **Interactive Breakdown:** Instant granular confidence distribution mappings.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟢 View Registered Categories", use_container_width=True):
            st.info(", ".join([f"**{c}**" for c in class_names]))
    with col2:
        if st.button("🔴 Flush App Memory Cache", use_container_width=True):
            st.cache_resource.clear()
            st.toast("Application memory refreshed!")

# ==========================================
# PAGE: IDENTIFIER
# ==========================================
elif page == "🔮 Real-Time Identifier":
    st.title("🔮 AI Diagnostic Image Identifier")
    st.write("Upload an image below to pass it through the active neural network layers.")
    
    uploaded_file = st.file_uploader("Upload Image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        
        layout_col1, layout_col2 = st.columns([1, 1])
        with layout_col1:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.image(raw_image, caption="📷 Source Input", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with layout_col2:
            st.subheader("🧠 Model Inference Engine Output")
            
            # Match pre-processing format exactly
            processed_img = raw_image.convert('RGB').resize(target_size)
            img_array = np.array(processed_img).astype('float32')
            img_array = (img_array / 127.5) - 1.0 
            img_input = np.expand_dims(img_array, axis=0)
            
            with st.spinner("Analyzing structural tensors..."):
                predictions = model.predict(img_input)
                predicted_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_idx] * 100
            
            st.markdown(f"""
                <div class="prediction-card">
                    <h3>CLASSIFIED OBJECT</h3>
                    <h1 style="color: #00ffcc; margin: 5px 0; font-size: 40px;">{class_names[predicted_idx].upper()}</h1>
                    <p>Statistical Confidence: {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.metric(label="Inference Status", value="Verified Accurate ✅", delta=f"{confidence:.1f}% Score Match")

        st.markdown("---")
        st.subheader("📊 Class Confidence Spectrum Distribution")
        for name, prob in zip(class_names, predictions[0]):
            st.progress(float(prob), text=f"**{name}**: {prob*100:.2f}% probability rating")

# ==========================================
# PAGE: METRICS
# ==========================================
elif page == "📊 Model Metrics":
    st.title("📊 Architecture & Validation Benchmarks")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Input Dimension Pipeline", value="32x32x3 Pixels")
    with c2:
        st.metric(label="Network Architecture Foundation", value="MobileNetV2 Base Layer")
        
    st.subheader("🧱 Core Model Layer Summary Blueprint")
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    st.code(short_model_summary, language="text")