import streamlit as st
import joblib
import pandas as pd
import os
from train_model import train_model

def prediction():
    st.title("PREDIKSI HARGA VILA")
    
    
    # Load model dan data
    model = joblib.load('model_regresi.pkl')
    le = joblib.load('location_encoder.pkl')
    
    # Input form
    tamu = st.number_input("JUMLAH TAMU", min_value=1, value=2)
    kamar = st.number_input("JUMLAH KAMAR TIDUR", min_value=1, value=1)
    tempat_tidur = st.number_input("JUMLAH TEMPAT TIDUR", min_value=1, value=1)
    lokasi = st.selectbox("LOKASI", options=le.classes_)
    
    if st.button("PREDIKSI"):
        lokasi_encoded = le.transform([lokasi])[0]
        input_data = [[tamu, kamar, tempat_tidur, lokasi_encoded]]
        harga_pred = model.predict(input_data)[0]
        st.success(f"HARGA PREDIKSI: ¥ {harga_pred:,.0f}".replace(",", "."))