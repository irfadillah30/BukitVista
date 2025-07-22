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

    data, location_encoder, lokasi_list = load_data()

    # Input lokasi cuma nama asli yang user lihat
    location = st.selectbox('Lokasi', lokasi_list)
    location_encoded = location_encoder.transform([location])[0]

    # Input fitur lain
    guests = st.slider('Jumlah Tamu', 1, 16, 2)
    bedrooms = st.slider('Jumlah Kamar Tidur', 1, 10, 1)
    beds = st.slider('Jumlah Tempat Tidur', 1, 15, 1)

    input_data = np.array([[guests, bedrooms, beds, location_encoded]])
    st.write("### Input Data untuk Prediksi")
    st.write(pd.DataFrame(input_data, columns=['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded']))  
    st.write("### Data yang digunakan untuk prediksi:")
    st.write(data.head())
    st.write("### Lokasi yang tersedia:")
    st.write(lokasi_list)
    st.write("### Fitur yang digunakan:")
    st.write(['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded'])
    st.write("### Model yang digunakan:")
    st.write("Random Forest Regressor")
    st.write("### Proses:")
    

