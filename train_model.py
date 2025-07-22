import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from data_loader import load_data
import streamlit as st
import pandas as pd


def train_model():
    data, location_encoder = load_data()

    st.write("### Debug Info: Data sebelum bersih")
    st.write(data.head())
    st.write(data.isnull().sum())

    # Pastikan kolom Harga numerik
    data['Harga'] = pd.to_numeric(data['Harga'], errors='coerce')

    # Buang baris yang ada NaN di fitur atau target
    data_clean = data.dropna(subset=['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded', 'Harga'])

    st.write("### Debug Info: Data setelah dropna")
    st.write(data_clean.head())
    st.write(data_clean.isnull().sum())
    st.write(f"Jumlah data setelah bersih: {len(data_clean)}")

    if len(data_clean) < 10:
        st.error("‼️ Data terlalu sedikit setelah pembersihan, training tidak bisa dilakukan.")
        return None, None, None, None

    X = data_clean[['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded']]
    y = data_clean['Harga']

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError as e:
        st.error(f"Error saat split data: {e}")
        return None, None, None, None

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Error saat training model: {e}")
        return None, None, None, None

    # Simpan model dan encoder
    joblib.dump(model, 'model_regresi.pkl')
    joblib.dump(location_encoder, 'location_encoder.pkl')

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.write("✅ Model dan encoder disimpan.")
    st.write(f"🔍 RMSE: {rmse:.2f}")
    st.write(f"🔍 R² Score: {r2:.2f}")

    return model, X_test, y_test, X
