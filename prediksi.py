import streamlit as st
import numpy as np
from train_model import train_model
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

def prediction():
    st.title('🏝️ Prediksi Harga Vila di Bali')

    # Ambil model dan data
    model, X_test, y_test, le, lokasi_list = train_model()

    # Input dari pengguna
    guests = st.slider('Jumlah Tamu', 1, 16, 2)
    bedrooms = st.slider('Jumlah Kamar Tidur', 1, 10, 1)
    beds = st.slider('Jumlah Tempat Tidur', 1, 15, 1)
    lokasi = st.selectbox('Lokasi', lokasi_list)

    # Encode lokasi
    lokasi_encoded = le.transform([lokasi])[0]

    # Prediksi harga
    input_data = np.array([[guests, bedrooms, beds, lokasi_encoded]])
    prediksi_harga = model.predict(input_data)[0]

    # Tampilkan hasil prediksi
    harga_formatted = f"¥{int(round(prediksi_harga)):,}".replace(",", ".")
    st.subheader('💰 Prediksi Harga')
    st.success(harga_formatted)

    # Evaluasi model
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader('📊 Skor Evaluasi')
    st.write(f'RMSE: ¥{rmse:,.0f}'.replace(",", "."))
    st.write(f'R²: {r2:.2f}')
