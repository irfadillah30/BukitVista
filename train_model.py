from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_loader import load_data

def train_model():
    data, le, lokasi_list = load_data()

    X = data[['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Lokasi Encoded']]
    y = data['Harga']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, le, lokasi_list
