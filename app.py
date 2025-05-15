import streamlit as st
import pandas as pd
from datetime import datetime
import pickle

# Load saved models
with open('score_pipeline.pkl', 'rb') as f:
    score_pipeline = pickle.load(f)

with open('result_pipeline.pkl', 'rb') as f:
    result_pipeline = pickle.load(f)

# Load trained models and preprocessing pipeline
score_model = pickle.load(open("score_pipeline.pkl", "rb"))
result_model = pickle.load(open("result_pipeline.pkl", "rb"))

st.set_page_config(page_title="Virat Kohli Match Predictor", layout="centered")

st.title("🏏 Virat Kohli Century & Match Win Predictor")
st.markdown("Predict if Virat will score a **century** and if **India will win** the match!")

# Input fields
opponent = st.selectbox("Opponent", ['Australia', 'England', 'Pakistan', 'New Zealand', 'South Africa', 'Sri Lanka', 'Bangladesh'])
venue = st.selectbox("Venue", ['Wankhede Stadium', 'Melbourne Cricket Ground', 'Eden Gardens', 'Dubai International Cricket Stadium'])
format_type = st.selectbox("Match Format", ['ODI', 'T20I', 'Test'])
batting_order = st.slider("Batting Position", 1, 6, 3)
innings = st.selectbox("Innings", [1, 2])
is_captain = st.checkbox("Is Kohli the Captain?", True)
is_home = st.checkbox("Is it a Home Match?", True)
is_mom = st.checkbox("Will he be Man of the Match?", False)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'Batting Order': [batting_order],
        'Inn.': [innings],
        'Against': [opponent],
        'Venue': [venue],
        'Format': [format_type],
        'Year': [datetime.now().year],
        'Month': [datetime.now().month],
        'Captain?': [1 if is_captain else 0],
        'Home?': [1 if is_home else 0],
        'Notout': [0],
        'MOM?': [1 if is_mom else 0]
    })

    predicted_score = score_model.predict(input_data)[0]
    predicted_result = result_model.predict(input_data)[0]

    st.subheader("📊 Prediction Results")
    
    # Display prediction results as simple text
    st.write(f"**Predicted Score**: {round(predicted_score)} runs")
    if predicted_score >= 100:
        st.success("💯 Century predicted!")
    else:
        st.warning("No Century likely.")
    
    st.write(f"**Match Result**: {'India Wins' if predicted_result == 1 else 'India May Not Win'}")