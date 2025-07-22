from sklearn.preprocessing import LabelEncoder
import pandas as pd
from utils import extract_facilities

def load_data(path='Daftar_vila_Dibali_Bukit_Vista_Cleaned.csv'):
    data = pd.read_csv(path)

   
    facilities = data['Fasilitas'].apply(extract_facilities)
    data['Jumlah Tamu'] = facilities.apply(lambda x: x[0])
    data['Jumlah Kamar Tidur'] = facilities.apply(lambda x: x[1])
    data['Jumlah Tempat Tidur'] = facilities.apply(lambda x: x[2])
    data['Jumlah Kamar Mandi'] = facilities.apply(lambda x: x[3])

   
    le = LabelEncoder()
    data['Lokasi Encoded'] = le.fit_transform(data['Lokasi'])

    return data, le
