import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_data
from train_model import train_model

def prediction():
    st.title('Prediksi Harga Vila di Bali')

    data, location_encoder = load_data()
    model, X_test, y_test, X = train_model()

    # Langsung tampil input dan prediksi tanpa cek model None
    guests = st.slider('Jumlah Tamu', 1, 16, 2)
    bedrooms = st.slider('Jumlah Kamar Tidur', 1, 10, 1)
    beds = st.slider('Jumlah Tempat Tidur', 1, 15, 1)
    location = st.selectbox('Lokasi', location_encoder.classes_)
    location_encoded = location_encoder.transform([location])[0]

    input_data = np.array([[guests, bedrooms, beds, location_encoded]])

    try:
        prediction = model.predict(input_data)
        price = int(round(prediction[0]))
        st.subheader('Hasil Prediksi')
        st.success(f'￥{price:,}')
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")

    try:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.subheader('Evaluasi Model')
        st.write(f'RMSE: {rmse:,.2f}')
        st.write(f'R-squared: {r2:.2f}')
    except Exception as e:
        st.warning(f"Gagal menghitung evaluasi model: {e}")
