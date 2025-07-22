import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import extract_facilities

def load_data(path='Daftar_vila_Dibali_Bukit_Vista_Cleaned.csv'):
    data = pd.read_csv(path)
    data = data.dropna(subset=['Fasilitas', 'Lokasi', 'Harga'])

    # Pastikan hasil ekstrak lengkap dan valid
    data['Jumlah Tamu'] = facilities.apply(lambda x: x[0] if x and len(x) > 0 else 0)
    data['Jumlah Kamar Tidur'] = facilities.apply(lambda x: x[1] if x and len(x) > 1 else 0)
    data['Jumlah Tempat Tidur'] = facilities.apply(lambda x: x[2] if x and len(x) > 2 else 0)
    data['Jumlah Kamar Mandi'] = facilities.apply(lambda x: x[3] if x and len(x) > 3 else 0)

    # Encode kolom Lokasi
    lokasi_list = sorted(data['Lokasi'].unique()) 
    le = LabelEncoder()
    data['Lokasi Encoded'] = le.fit_transform(data['Lokasi'])

    return data, le, lokasi_list
