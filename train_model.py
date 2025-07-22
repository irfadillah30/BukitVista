import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from data_loader import load_data


def train_model():
    data, location_encoder = load_data()

    # Tangani missing values di fitur dan target
    data = data.dropna(subset=['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded', 'Harga'])

    X = data[['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded']]
    y = data['Harga']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Simpan model dan encoder
    joblib.dump(model, 'model_regresi.pkl')
    joblib.dump(location_encoder, 'location_encoder.pkl')

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("✅ Model dan encoder disimpan.")
    print(f"🔍 RMSE: {rmse:.2f}")
    print(f"🔍 R² Score: {r2:.2f}")

    return model, X_test, y_test, X


if __name__ == '__main__':
    train_model()