import streamlit as st
import pandas as pd

st.set_page_config(page_title="Diabetes Feature Engineering App", page_icon="🩺", layout="centered")

st.title("Diabetes  Uygulaması")
st.write("Bu uygulama kullanıcı girdilerine göre yeni feature'lar üretir.")

st.subheader("Kullanıcı Bilgileri")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
blood_pressure = st.number_input("BloodPressure", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("SkinThickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0)
dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.50)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Feature'ları Oluştur"):
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    df = pd.DataFrame([data])

    # NEW_AGE_CAT
    if df.loc[0, "Age"] < 35:
        df.loc[0, "NEW_AGE_CAT"] = "young"
    elif 35 <= df.loc[0, "Age"] < 55:
        df.loc[0, "NEW_AGE_CAT"] = "middleage"
    else:
        df.loc[0, "NEW_AGE_CAT"] = "old"

    # NEW_BMI_CAT
    if df.loc[0, "BMI"] < 18.5:
        df.loc[0, "NEW_BMI_CAT"] = "underweight"
    elif 18.5 <= df.loc[0, "BMI"] < 24.9:
        df.loc[0, "NEW_BMI_CAT"] = "normal"
    elif 25 <= df.loc[0, "BMI"] < 29.9:
        df.loc[0, "NEW_BMI_CAT"] = "overweight"
    else:
        df.loc[0, "NEW_BMI_CAT"] = "obese"

    # NEW_GLUCOSE_CAT
    if df.loc[0, "Glucose"] < 70:
        df.loc[0, "NEW_GLUCOSE_CAT"] = "low"
    elif 70 <= df.loc[0, "Glucose"] < 100:
        df.loc[0, "NEW_GLUCOSE_CAT"] = "normal"
    elif 100 <= df.loc[0, "Glucose"] < 126:
        df.loc[0, "NEW_GLUCOSE_CAT"] = "prediabetes"
    else:
        df.loc[0, "NEW_GLUCOSE_CAT"] = "diabetes_risk"

    # NEW_AGE_BMI
    if df.loc[0, "Age"] < 35 and df.loc[0, "BMI"] < 30:
        df.loc[0, "NEW_AGE_BMI"] = "young_normal"
    elif df.loc[0, "Age"] < 35 and df.loc[0, "BMI"] >= 30:
        df.loc[0, "NEW_AGE_BMI"] = "young_obese"
    elif 35 <= df.loc[0, "Age"] < 55 and df.loc[0, "BMI"] < 30:
        df.loc[0, "NEW_AGE_BMI"] = "middleage_normal"
    elif 35 <= df.loc[0, "Age"] < 55 and df.loc[0, "BMI"] >= 30:
        df.loc[0, "NEW_AGE_BMI"] = "middleage_obese"
    elif df.loc[0, "Age"] >= 55 and df.loc[0, "BMI"] < 30:
        df.loc[0, "NEW_AGE_BMI"] = "old_normal"
    else:
        df.loc[0, "NEW_AGE_BMI"] = "old_obese"

    # Sayısal yeni feature'lar
    df["NEW_AGE_GLUCOSE"] = df["Age"] * df["Glucose"]
    df["NEW_BMI_GLUCOSE"] = df["BMI"] * df["Glucose"]

    st.subheader("Oluşturulan Yeni Feature'lar")
    st.dataframe(df[[
        "NEW_AGE_CAT",
        "NEW_BMI_CAT",
        "NEW_GLUCOSE_CAT",
        "NEW_AGE_BMI",
        "NEW_AGE_GLUCOSE",
        "NEW_BMI_GLUCOSE"
    ]])

    st.subheader("Kısa Yorum")
    st.write(f"Yaş kategorisi: **{df.loc[0, 'NEW_AGE_CAT']}**")
    st.write(f"BMI kategorisi: **{df.loc[0, 'NEW_BMI_CAT']}**")
    st.write(f"Glikoz kategorisi: **{df.loc[0, 'NEW_GLUCOSE_CAT']}**")
    st.write(f"Yaş-BMI profili: **{df.loc[0, 'NEW_AGE_BMI']}**")
    st.write(f"Age x Glucose: **{df.loc[0, 'NEW_AGE_GLUCOSE']:.2f}**")
    st.write(f"BMI x Glucose: **{df.loc[0, 'NEW_BMI_GLUCOSE']:.2f}**")