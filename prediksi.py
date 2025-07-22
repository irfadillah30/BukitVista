import streamlit as st
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def prediction():
    st.title('Prediksi Harga Vila di Bali')

    
    model = joblib.load('model_regresi.pkl')
    location_encoder = joblib.load('location_encoder.pkl')
    lokasi_list = list(location_encoder.classes_)

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

if __name__ == "__main__":
    prediction()
