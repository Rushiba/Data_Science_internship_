import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Page Config
st.set_page_config(page_title="Iris Classification", layout="wide")

# 2. THE PERFECT CSS: Huge Fonts & Shining Buttons
st.markdown("""
    <style>
    /* Forced High-Res Flower Background */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                    url("https://images.unsplash.com/photo-1470770841072-f978cf4d019e?q=80&w=2070&auto=format&fit=crop") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }

    /* Massive Glowing Header */
    h1 {
        font-size: 90px !important;
        color: #00FF7F !important;
        text-shadow: 0 0 30px #00FF7F, 0 0 60px #00D2FF !important;
        text-align: center;
        font-weight: 900 !important;
    }

    /* Huge Slider Labels & Text */
    .stSlider label, p, h2, h3 {
        font-size: 28px !important;
        color: white !important;
        font-weight: bold !important;
        text-shadow: 2px 2px 5px black !important;
    }

    /* Glassmorphism Input Box */
    .glass-box {
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 30px;
        padding: 40px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 0 40px rgba(0,0,0,0.8);
    }

    /* SHINING INTERACTIVE BUTTONS */
    .stButton > button {
        background: linear-gradient(45deg, #FF00CC, #3333FF) !important;
        color: white !important;
        font-size: 35px !important;
        height: 100px !important;
        width: 100% !important;
        border-radius: 50px !important;
        border: 4px solid #FFFFFF !important;
        box-shadow: 0 0 25px #FF00CC !important;
        transition: 0.4s ease !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 60px #00D2FF !important;
        border-color: #00D2FF !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model with Scaler
@st.cache_resource
def load_all():
    try:
        m = joblib.load('iris_model.joblib')
        l = joblib.load('label_encoder.joblib')
        s = joblib.load('scaler.joblib')
        return m, l, s
    except: return None, None, None

model, le, scaler = load_all()

# 4. Website Body
st.markdown("<h1>🌸 Iris Classification 🌸</h1>", unsafe_allow_html=True)

if model is None:
    st.error("🚨 ERROR: Please run 'python train.py' first to create your model files!")
else:
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.subheader("⚙️ Biological Settings")
        sl = st.slider("Sepal Length", 4.0, 8.0, 5.8)
        sw = st.slider("Sepal Width", 2.0, 4.5, 3.0)
        pl = st.slider("Petal Length", 1.0, 7.0, 4.3)
        pw = st.slider("Petal Width", 0.1, 2.5, 1.3)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("✨ ANALYZE NOW"):
            # Prepare and Scale data
            raw_input = pd.DataFrame([[sl, sw, pl, pw]], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
            scaled_input = scaler.transform(raw_input)
            
            # Prediction
            pred = model.predict(scaled_input)[0]
            species = le.inverse_transform([pred])[0]
            
            st.markdown(f"""
                <div style="background: rgba(0, 255, 127, 0.2); border: 4px solid #00FF7F; padding: 25px; border-radius: 20px; text-align: center;">
                    <h2 style="color: #00FF7F !important; font-size: 50px; margin:0;">RESULT: {species.upper()}</h2>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader( "Neural Map")
        df = pd.read_csv('Iris.csv')
        fig = px.scatter_3d(df, x='SepalLengthCm', y='PetalLengthCm', z='PetalWidthCm',
                            color='Species', color_discrete_sequence=["#FF00CC", "#00D2FF", "#00FF7F"],
                            template="plotly_dark")
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            scene=dict(bgcolor='rgba(0,0,0,0.4)'),
            font=dict(size=16, color="white")
        )
        st.plotly_chart(fig, use_container_width=True)