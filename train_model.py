import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from data_loader import load_data

def train_model():
    # Load data
    data, le, _ = load_data()
    
    # Siapkan features dan target
    features = ['Jumlah_Tamu', 'Jumlah_Kamar_Tidur', 'Jumlah_Tempat_Tidur', 'Lokasi Encoded']
    X = data[features]
    y = data['Harga']
    
    # Handle missing values di features
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model dan imputer
    joblib.dump(model, 'model_regresi.pkl')
    joblib.dump(le, 'location_encoder.pkl')
    joblib.dump(imputer, 'feature_imputer.pkl')
    
    return model, X_test, y_test, X