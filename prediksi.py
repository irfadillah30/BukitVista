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

    # Load data dan model
    data, location_encoder = load_data()
    model, X_test, y_test, X = train_model()

    if model is None:
        st.error("Model belum tersedia karena error saat training.")
        return

    st.header('Input Fitur')
    guests = st.slider('Jumlah Tamu', 1, 16, 2)
    bedrooms = st.slider('Jumlah Kamar Tidur', 1, 10, 1)
    beds = st.slider('Jumlah Tempat Tidur', 1, 15, 1)
    location = st.selectbox('Lokasi', location_encoder.classes_)
    location_encoded = location_encoder.transform([location])[0]

    # Buat input data bentuk array 2D
    input_data = np.array([[guests, bedrooms, beds, location_encoded]])

    # Prediksi harga vila
    prediction = model.predict(input_data)

    st.subheader('Hasil Prediksi')
    st.success(f'￥{int(round(prediction[0])):,}')

    # Evaluasi model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader('Evaluasi Model')
    st.write(f'RMSE: {rmse:,.2f}')
    st.write(f'R-squared: {r2:.2f}')

    # Visualisasi feature importance
    if hasattr(model, 'feature_importances_'):
        st.subheader('Pentingnya Fitur')

        fitur = X.columns if isinstance(X, pd.DataFrame) else ['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi']

        feature_importance = pd.DataFrame({
            'Fitur': fitur,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Importance', y='Fitur', data=feature_importance, ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)
    else:
        st.write('Model tidak memiliki atribut feature_importances_ untuk menampilkan pentingnya fitur.')
