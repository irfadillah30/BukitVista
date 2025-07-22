import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def prediction():
    st.title("PREDIKSI HARGA VILA")
    
    # Load model dan data test
    model = joblib.load('model_regresi.pkl')
    le = joblib.load('location_encoder.pkl')
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
    
    # Hitung metrik
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Input form
    st.header("Input Data Vila")
    tamu = st.number_input("JUMLAH TAMU", min_value=1, value=2)
    kamar = st.number_input("JUMLAH KAMAR TIDUR", min_value=1, value=1)
    tempat_tidur = st.number_input("JUMLAH TEMPAT TIDUR", min_value=1, value=1)
    lokasi = st.selectbox("LOKASI", options=le.classes_)
    
    if st.button("PREDIKSI"):
        # Encode lokasi
        lokasi_encoded = le.transform([lokasi])[0]
        
        # Predict
        input_data = [[tamu, kamar, tempat_tidur, lokasi_encoded]]
        harga_pred = model.predict(input_data)[0]
        
        # Tampilkan hasil
        st.success(f"**HARGA PREDIKSI:** ¥ {harga_pred:,.0f}".replace(",", "."))
        st.write(f"**RMSE:** {rmse:,.0f}")
        st.write(f"**R²:** {r2:.3f}")