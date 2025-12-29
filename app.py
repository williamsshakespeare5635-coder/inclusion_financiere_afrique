import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.title("Prédiction de l'Inclusion Financière")

# Création du formulaire
user_input = {}

for feature in features:
    user_input[feature] = st.number_input(feature, value=0)

if st.button("Prédire"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Cette personne possède / utilisera un compte bancaire")
    else:
        st.error("❌ Cette personne ne possède pas de compte bancaire")
