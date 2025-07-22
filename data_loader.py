import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import extract_facilities

def load_data(path='Daftar_vila_Dibali_Bukit_Vista_Cleaned.csv'):
    data = pd.read_csv(path)


    # Pastikan ekstraksi fasilitas aman
    facilities = data['Fasilitas'].apply(extract_facilities)
    data['Jumlah Tamu'] = facilities.apply(lambda x: x[0])
    data['Jumlah Kamar Tidur'] = facilities.apply(lambda x: x[1])
    data['Jumlah Tempat Tidur'] = facilities.apply(lambda x: x[2])
    data['Jumlah Kamar Mandi'] = facilities.apply(lambda x: x[3])


    # Label encode Lokasi
    le = LabelEncoder()
    data['Lokasi Encoded'] = le.fit_transform(data['Lokasi'])
    lokasi_list = sorted(data['Lokasi'].unique())

    # Bersihkan kolom Harga dari simbol ¥, titik, spasi
    data['Harga'] = (
        data['Harga']
        .astype(str)
        .str.replace(r'[¥¥,\s]', '', regex=True)
        .str.replace('.', '', regex=False)
        .replace('', None)
        .astype(float)
    )

    # Isi NaN dengan median harga
    data['Harga'] = data['Harga'].fillna(data['Harga'].median()).astype(int)

    return data, le, lokasi_list
