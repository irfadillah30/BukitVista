import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from data_loader import load_data

def train_model():
    data, location_encoder, _ = load_data()

    data['Harga'] = pd.to_numeric(data['Harga'], errors='coerce')

    for col in ['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Harga']:
        data[col].fillna(data[col].mean(), inplace=True)

    X = data[['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded']]
    y = data['Harga']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Evaluasi model: RMSE = {rmse:.2f}, R² = {r2:.2f}")

    # Simpan model dan encoder
    joblib.dump(model, 'model_regresi.pkl')
    joblib.dump(location_encoder, 'location_encoder.pkl')

    print("Model dan encoder berhasil disimpan.")

    return model

if __name__ == "__main__":
    train_model()
