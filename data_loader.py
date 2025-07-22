import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import extract_facilities

def load_data(path='Daftar_vila_Dibali_Bukit_Vista_Cleaned.csv'):
    # Load data
    data = pd.read_csv(path)
    
    # Menghapus simbol "¥" dan titik, lalu konversi ke angka
    data['Harga'] = data['Harga'].astype(str)
    data['Harga'] = data['Harga'].str.replace('¥', '', regex=False).str.replace('.', '', regex=False).str.strip()

    # Isi NaN dengan median sebelum konversi ke int
    data['Harga'] = data['Harga'].replace('', np.nan)
    data['Harga'] = data['Harga'].astype(float)
    median_harga = data['Harga'].median()
    data['Harga'] = data['Harga'].fillna(median_harga).astype(int)
    
    # Ekstrak fasilitas
    facilities = data['Fasilitas'].apply(extract_facilities)
    data['Jumlah_Tamu'] = data['Fasilitas'].str.extract(r'(\d+)\s*tamu').astype(float)
    data['Jumlah_Kamar'] = data['Fasilitas'].str.extract(r'(\d+)\s*kamar tidur').astype(float)
    data['Jumlah_Tempat_Tidur'] = data['Fasilitas'].str.extract(r'(\d+)\s*tempat tidur').astype(float)
    
    # Encode lokasi
    le = LabelEncoder()
    data['Lokasi_Encoded'] = le.fit_transform(data['Lokasi'])
    
    return data, le, le.classes_.tolist()