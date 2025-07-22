import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_data

def train_model():
    data, location_encoder, _ = load_data()

    # Pastikan kolom numerik dan buat kolom fitur
    data['Harga'] = pd.to_numeric(data['Harga'], errors='coerce')

    # Isi nilai kosong dengan rata-rata kolom
    for col in ['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Harga']:
        data[col].fillna(data[col].mean(), inplace=True)
    data['Lokasi Encoded'] = location_encoder.transform(data['Lokasi'])

    # Fitur dan target
    X = data[['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded']]
    y = data['Harga']

    # Split data (boleh disimpan kalau ingin evaluasi di training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Simpan model dan encoder
    joblib.dump(model, 'model_regresi.pkl')
    joblib.dump(location_encoder, 'location_encoder.pkl')

    print("Model dan encoder berhasil disimpan.")

    return model  # Kembalikan cuma model saja, ga perlu X_test atau y_test

if __name__ == "__main__":
    train_model()
