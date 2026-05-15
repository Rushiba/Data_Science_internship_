import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="HealthCalc AI", layout="wide", page_icon="🚀")

# --- PREMIUM CSS CORE ---
st.markdown(r"""
    <style>
    /* 1. ANIMATED BACKGROUND */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #00d2ff);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient { 0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;} }

    /* 2. SHINING SIDEBAR CARDS */
    [data-testid="stSidebar"] [role="radiogroup"] { display: flex; flex-direction: column; gap: 12px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        width: 100% !important;
        transition: all 0.4s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label div[data-testid="stWidgetLabel"] div:first-child { display: none !important; }
    
    [data-testid="stSidebar"] [role="radiogroup"] label::after {
        content: ""; position: absolute; top: 0; left: -150%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: 0.6s;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover::after { left: 150%; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { transform: translateX(10px) !important; background: rgba(138, 43, 226, 0.3) !important; }

    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(90deg, #8A2BE2, #EE82EE) !important;
        box-shadow: 0px 0px 20px rgba(238, 130, 238, 0.8) !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] p { color: white !important; font-weight: bold !important; }

    /* 3. UNIVERSAL GLASS STYLING */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }
    .shimmer-text {
        background: linear-gradient(90deg, #fff, #EE82EE, #8A2BE2, #fff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        font-weight: 800;
        font-size: 3rem !important;
    }
    @keyframes shine { to { background-position: 200% center; } }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('insurance_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except: return None, None

model, scaler = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#EE82EE; text-align:center;'>HEALTH AI</h2>", unsafe_allow_html=True)
    page = st.radio("NAV", ["🏠 Home", "ℹ️ About", "🔮 Prediction", "📊 Results"], label_visibility="collapsed")
    st.divider()
    st.info("Precision Analytics Engine")

# --- PAGE: HOME ---
if page == "🏠 Home":
    st.markdown('<h1 class="shimmer-text">HealthCalc AI</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h2 style="color:#EE82EE;">AI-Powered Insurance Estimator</h2>
            <p style="font-size:1.15rem;">HealthCalc AI uses <b>Linear Regression</b> to provide accurate predictions of annual medical insurance premiums. By analyzing key demographic and lifestyle data, our system helps you plan your healthcare budget with precision.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### How It Works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="glass-card" style="padding:15px; text-align:center;"><h4>1. Input Data</h4><p style="font-size:0.9rem;">Provide your age, BMI, and lifestyle habits.</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card" style="padding:15px; text-align:center;"><h4>2. AI Analysis</h4><p style="font-size:0.9rem;">Our model processes your data against 6 key variables.</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="glass-card" style="padding:15px; text-align:center;"><h4>3. Get Result</h4><p style="font-size:0.9rem;">Receive an instant annual cost estimate.</p></div>', unsafe_allow_html=True)

    with col2:
        img_path = "artificial-intelligence-in-healthcare-concept-with-person-interacting-with-ai-powered-devices-and-receiving-diagnosis-vector.jpg"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.markdown("""
        <div class="glass-card" style="margin-top:10px;">
            <h4 style="color:#EE82EE;">Quick Stats</h4>
            <ul style="font-size:0.9rem;">
                <li>Trained on 1,300+ medical records</li>
                <li>Standardized Feature Scaling</li>
                <li>Instant Prediction Engine</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- PAGE: ABOUT ---
elif page == "ℹ️ About":
    st.markdown('<h1 class="shimmer-text">Project Intelligence</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3>System Overview</h3>
        <p>This application identifies correlations between lifestyle choices and financial risk. Using a <b>Linear Regression</b> model, we calculate how variables like smoking status and BMI disproportionately affect healthcare costs.</p>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#EE82EE;">Variable Influence</h4>
            <ul>
                <li><b>Smoker Status:</b> The highest weighted feature in our model.</li>
                <li><b>BMI:</b> Used to assess physical risk factors.</li>
                <li><b>Regional Data:</b> Adjusts for varying healthcare costs across 4 US regions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#EE82EE;">Development Stack</h4>
            <ul>
                <li><b>Algorithm:</b> Linear Regression (Scikit-Learn)</li>
                <li><b>Preprocessing:</b> StandardScaler (6-feature input)</li>
                <li><b>Interface:</b> Streamlit with custom CSS glassmorphism</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- PAGE: PREDICTION ---
elif page == "🔮 Prediction":
    st.title("User Profile Analysis")
    if model is None: st.error("Assets not found!")
    else:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", 18, 100, 25)
                sex = st.selectbox("Sex", ["male", "female"])
                bmi = st.slider("BMI", 10.0, 50.0, 24.0)
            with c2:
                children = st.number_input("Children", 0, 10, 0)
                smoker = st.selectbox("Smoker?", ["no", "yes"])
                region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
            
            if st.button("✨ GENERATE ESTIMATE"):
                sex_val = 1 if sex == "male" else 0
                smoker_val = 1 if smoker == "yes" else 0
                reg_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
                # FIXED: EXACTLY 6 FEATURES FOR THE SCALER
                input_data = np.array([[age, sex_val, bmi, children, smoker_val, reg_map[region]]])
                scaled = scaler.transform(input_data)
                prediction = model.predict(scaled)[0]
                st.session_state['pred'] = prediction
                st.success("Analysis Complete! Go to Results.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: RESULTS ---
elif page == "📊 Results":
    st.title("Final Analysis")
    if 'pred' in st.session_state:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <h3>Calculated Annual Premium</h3>
            <h1 style="color:#EE82EE;">${st.session_state['pred']:,.2f}</h1>
            <p>Our AI recommends reviewing your insurance plan if this estimate differs significantly from your current premium.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please complete a prediction first.")