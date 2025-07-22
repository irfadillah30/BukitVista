import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import extract_facilities

def load_data(path='Daftar_vila_Dibali_Bukit_Vista_Cleaned.csv'):
    # Load data
    data = pd.read_csv(path)
    
    # Bersihkan harga
    data['Harga'] = data['Harga'].str.replace('¥', '').str.replace('.', '').str.strip()
    data['Harga'] = pd.to_numeric(data['Harga'], errors='coerce')
    
    # Isi missing values dengan rata-rata
    data['Harga'] = data['Harga'].fillna(data['Harga'].mean())
    
    # Ekstrak fasilitas
    facilities = data['Fasilitas'].apply(lambda x: extract_facilities(x) if pd.notnull(x) else (1, 1, 1))  # Default 1 jika kosong
    data['Jumlah Tamu'] = [f[0] for f in facilities]
    data['Jumlah Kamar Tidur'] = [f[1] for f in facilities]
    data['Jumlah Tempat Tidur'] = [f[2] for f in facilities]
    
    # Encode lokasi
    le = LabelEncoder()
    data['Lokasi'] = data['Lokasi'].fillna('Unknown')
    data['Lokasi Encoded'] = le.fit_transform(data['Lokasi'])
    
    return data, le, le.classes_.tolist()