import streamlit as st
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from data_loader import load_data

st.markdown("""
    <style>
    .stButton > button {
        background-color: #d9534f;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def prediction():
    st.title('Prediksi Harga Vila di Bali')

    model = joblib.load('model_regresi.pkl')
    location_encoder = joblib.load('location_encoder.pkl')
    lokasi_list = list(location_encoder.classes_)

    data, le, _ = load_data()
    X = data[['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded']]
    y = data['Harga']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    guests = st.slider('Jumlah Tamu', 1, 16, 2)
    bedrooms = st.slider('Jumlah Kamar Tidur', 1, 10, 1)
    beds = st.slider('Jumlah Tempat Tidur', 1, 15, 1)
    lokasi = st.selectbox('Lokasi', lokasi_list)
    lokasi_encoded = location_encoder.transform([lokasi])[0]

     # Tombol untuk prediksi
    if st.button('Prediksi 🔎'):
        input_data = np.array([[guests, bedrooms, beds, lokasi_encoded]])
        prediksi_harga = model.predict(input_data)

        st.subheader('Prediksi Harga 💴 ')
        harga_formatted = f"{int(round(prediksi_harga[0])):,} ¥".replace(',', '.')
        st.success(harga_formatted)


if __name__ == "__main__":
    prediction()