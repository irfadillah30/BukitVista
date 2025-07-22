import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_data

def train_model():
    data, location_encoder, _ = load_data()

    data['Harga'] = pd.to_numeric(data['Harga'], errors='coerce')

    data_clean = data.dropna(subset=[
        'Jumlah Tamu',
        'Jumlah Kamar Tidur',
        'Jumlah Tempat Tidur',
        'Lokasi Encoded',
        'Harga'
    ])

    if len(data_clean) < 10:
        return None, None, None, None

    X = data_clean[['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded']]
    y = data_clean['Harga']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Simpan model dan encoder
    joblib.dump(model, 'model_regresi.pkl')
    joblib.dump(location_encoder, 'location_encoder.pkl')

    return model, X_test, y_test, X
