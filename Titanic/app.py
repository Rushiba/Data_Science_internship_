import streamlit as st
import pandas as pd
import joblib

# 1. Page Config
st.set_page_config(page_title="Titanic Sea Voyage", page_icon="⚓", layout="centered")

# 2. Sea Theme CSS
def apply_theme(status=None):
    # Colors based on survival status
    if status == "survived":
        overlay = "rgba(40, 167, 69, 0.3)"  # Transparent Green
    elif status == "died":
        overlay = "rgba(220, 53, 69, 0.3)"   # Transparent Red
    else:
        overlay = "rgba(0, 50, 100, 0.5)"   # Deep Sea Blue Overlay

    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient({overlay}, {overlay}), 
                        url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1920&q=80");
            background-size: cover;
            background-attachment: fixed;
        }}
        
        /* Make all text white and bold for high visibility */
        h1, h2, h3, p, label, .stMarkdown {{
            color: white !important;
            text-shadow: 2px 2px 4px #000000;
            font-weight: bold !important;
        }}

        /* Style the input cards */
        .stNumberInput, .stSelectbox, .stSlider, .stRadio {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 10px;
            backdrop-filter: blur(5px);
        }}
        </style>
        """, unsafe_allow_html=True)

# 3. Load Model
try:
    model = joblib.load('titanic_model.joblib')
except:
    st.error("⚠️ Model file not found! Please run 'train_model.py' first.")

# 4. App Interface
apply_theme() # Start with blue theme

st.markdown("<h1 style='text-align: center;'>🚢 TITANIC SURVIVAL PREDICTOR ⚓</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter details below to see if the passenger survives the voyage.</p>", unsafe_allow_html=True)

# 5. Organized Input Sections
st.markdown("### 🎫 Ticket & Cabin Info")
col1, col2 = st.columns(2)
with col1:
    pclass = st.radio("Select Passenger Class 🎫", [1, 2, 3], help="1=First Class, 2=Second, 3=Third")
    fare = st.number_input("Ticket Fare ($) 💰", 0.0, 512.0, 32.0)

with col2:
    embarked = st.selectbox("Departure Port 🚩", ["S", "C", "Q"], 
                            format_func=lambda x: "Southampton" if x=="S" else "Cherbourg" if x=="C" else "Queenstown")
    sex = st.selectbox("Gender 👤", ["male", "female"])

st.markdown("### 🧑 Passenger Personal Details")
col3, col4 = st.columns(2)
with col3:
    age = st.slider("Passenger Age 🎂", 0, 100, 25)

with col4:
    sibsp = st.number_input("Siblings/Spouses Aboard 👫", 0, 8, 0)
    parch = st.number_input("Parents/Children Aboard 👨‍👩‍👧", 0, 6, 0)

# 6. Processing
sex_val = 0 if sex == "male" else 1
emb_map = {"S": 0, "C": 1, "Q": 2}
emb_val = emb_map[embarked]

st.markdown("---")

if st.button("🔮 PREDICT MY FATE", use_container_width=True):
    # Prepare data exactly as the model expects
    input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare, emb_val]], 
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        apply_theme("survived") # Change background to green
        st.balloons()
        st.success(f"## 🎉 SURVIVED!")
        st.markdown(f"### Probability of Survival: {prob:.1f}%")
    else:
        apply_theme("died") # Change background to red
        st.error(f"## 💀 DID NOT SURVIVE")
        st.markdown(f"### Probability of Survival: {prob:.1f}%")

    st.info(f"Summary: A {age} year old {sex} in class {pclass} paying ${fare:.2f} had a {prob:.1f}% chance of survival.")
