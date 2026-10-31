import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("🩺 Diabetes Risk Predictor")
st.caption("Educational tool only. Not a medical diagnosis.")

c1, c2 = st.columns(2)
preg = c1.number_input("Pregnancies", 0, 20, 1)
glu = c2.number_input("Glucose (mg/dL)", 40, 300, 120)
bp = c1.number_input("Blood Pressure (mm Hg)", 30, 200, 70)
skin = c2.number_input("Skin Thickness (mm)", 5, 100, 25)
ins = c1.number_input("Insulin (mu U/ml)", 10, 900, 80)
bmi = c2.number_input("BMI", 10.0, 70.0, 28.0)
dpf = c1.number_input("Diabetes Pedigree Function", 0.05, 2.5, 0.5)
age = c2.number_input("Age", 15, 100, 30)

if st.button("Predict"):
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    row = pd.DataFrame([[preg,glu,bp,skin,ins,bmi,dpf,age]], columns=cols)
    prob = model.predict_proba(scaler.transform(row))[0][1]
    if prob >= 0.5:
        st.error(f"High risk of diabetes (probability: {prob:.0%})")
    else:
        st.success(f"Low risk of diabetes (probability: {prob:.0%})")
    st.warning("This is not a medical diagnosis. Please consult a doctor.")
# paste the full app.py code from Step 6c here
