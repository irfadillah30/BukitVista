import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from data_loader import load_data  # Jika kamu punya fungsi ini

def prediction():
    st.title('Prediksi Harga Vila di Bali')

    # Load model dan encoder
    model = joblib.load('model_regresi.pkl')
    location_encoder = joblib.load('location_encoder.pkl')
    lokasi_list = list(location_encoder.classes_)

    # Load data asli untuk evaluasi (atau load test set kalau ada)
    data, le, _ = load_data()
    X = data[['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded']]
    y = data['Harga']

    # Split data supaya ada X_test, y_test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Input user
    guests = st.slider('Jumlah Tamu', 1, 16, 2)
    bedrooms = st.slider('Jumlah Kamar Tidur', 1, 10, 1)
    beds = st.slider('Jumlah Tempat Tidur', 1, 15, 1)
    lokasi = st.selectbox('Lokasi', lokasi_list)
    lokasi_encoded = location_encoder.transform([lokasi])[0]

    input_data = np.array([[guests, bedrooms, beds, lokasi_encoded]])
    prediksi_harga = model.predict(input_data)

    st.subheader('💰 Prediksi Harga')
    harga_formatted = f"¥{int(round(prediksi_harga[0])):,}".replace(',', '.')
    st.success(harga_formatted)

    # Evaluasi model di data test
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader('📊 Evaluasi Model')
    st.write(f'RMSE: {rmse:,.2f}')
    st.write(f'R²: {r2:.2f}')


if __name__ == "__main__":
    prediction()
