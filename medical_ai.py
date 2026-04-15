import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Medical Diagnosis", layout="centered")

# -------------------------------
# TRAIN MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model():
    data = {
        "fever": [1, 1, 0, 1, 0, 0, 1, 0],
        "cough": [1, 0, 1, 1, 0, 1, 1, 0],
        "headache": [1, 1, 0, 0, 1, 0, 1, 0],
        "fatigue": [1, 1, 0, 1, 0, 0, 1, 0],
        "disease": ["Flu", "Flu", "Cold", "Flu", "Healthy", "Cold", "Flu", "Healthy"]
    }

    df = pd.DataFrame(data)

    X = df[["fever", "cough", "headache", "fatigue"]]
    y = df["disease"]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    return model

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🩺 AI Medical Diagnosis System")

st.markdown("### Enter Symptoms")

fever = st.radio("Fever", ["No", "Yes"])
cough = st.radio("Cough", ["No", "Yes"])
headache = st.radio("Headache", ["No", "Yes"])
fatigue = st.radio("Fatigue", ["No", "Yes"])

# Convert input
def convert_input(val):
    return 1 if val == "Yes" else 0

input_data = [[
    convert_input(fever),
    convert_input(cough),
    convert_input(headache),
    convert_input(fatigue)
]]

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Disease"):

    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    confidence = max(probs) * 100

    # ---------------- RESULT ----------------
    st.subheader("📊 Result")

    if prediction == "Healthy":
        st.success("✅ You are likely Healthy")
    else:
        st.warning(f"⚠ Possible Disease: {prediction}")

    st.write(f"🔢 Confidence: {confidence:.2f}%")

    # ---------------- SUGGESTIONS ----------------
    st.subheader("💡 Suggestions")

    if prediction == "Flu":
        st.write("- Take proper rest")
        st.write("- Stay hydrated")
        st.write("- Consult doctor if symptoms worsen")

    elif prediction == "Cold":
        st.write("- Drink warm fluids")
        st.write("- Take steam inhalation")
        st.write("- Use basic medications")

    else:
        st.write("- Maintain a healthy lifestyle")