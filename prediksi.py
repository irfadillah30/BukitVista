import streamlit as st
import numpy as np
from data_loader import load_data
from train_model import train_model
from sklearn.metrics import mean_squared_error, r2_score

def prediction():
    st.title('Prediksi Harga Vila di Bali')

    data, location_encoder, lokasi_list = load_data()
    model, X_test, y_test, X = train_model()

    guests = st.slider('Jumlah Tamu', 1, 16, 2)
    bedrooms = st.slider('Jumlah Kamar Tidur', 1, 10, 1)
    beds = st.slider('Jumlah Tempat Tidur', 1, 15, 1)
    lokasi = st.selectbox('Lokasi', lokasi_list)
    lokasi_encoded = location_encoder.transform([lokasi])[0]

    input_data = np.array([[guests, bedrooms, beds, lokasi_encoded]])
    prediksi_harga = model.predict(input_data)

    st.subheader('Prediksi Harga')
    st.success(f'Rp {int(round(prediksi_harga[0])):,}')

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader('Skor Evaluasi')
    st.write(f'RMSE: {rmse:,.2f}')
    st.write(f'R-squared: {r2:.2f}')
