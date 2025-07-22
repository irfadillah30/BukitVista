import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import extract_facilities

def load_data(path='Daftar_vila_Dibali_Bukit_Vista_Cleaned.csv'):
    data = pd.read_csv(path)

    # Drop data yang kolom pentingnya kosong
    data = data.dropna(subset=['Fasilitas', 'Lokasi', 'Harga'])

    # Ekstrak fasilitas ke dalam kolom-kolom numerik
    facilities = data['Fasilitas'].apply(extract_facilities)

    data['Jumlah Tamu'] = facilities.apply(lambda x: x[0] if x and len(x) > 0 else 0)
    data['Jumlah Kamar Tidur'] = facilities.apply(lambda x: x[1] if x and len(x) > 1 else 0)
    data['Jumlah Tempat Tidur'] = facilities.apply(lambda x: x[2] if x and len(x) > 2 else 0)


    # Encode lokasi jadi angka (untuk model)
    lokasi_list = sorted(data['Lokasi'].unique())
    le = LabelEncoder()
    data['Lokasi Encoded'] = le.fit_transform(data['Lokasi'])

    return data, le, lokasi_list
