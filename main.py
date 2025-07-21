import streamlit as st
import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
st.set_page_config(layout='wide')
st.title('Analisis Fasilitas & Lokasi dengan Machine Learning')

st.sidebar.image('https://www.bukitvista.com/wp-content/uploads/2021/06/BukitVista-LOGO-ONLY-transparent.png', width=200)
st.header('Project Data Scientist')

st.sidebar.title('Model Prediksi Harga Vila')
page = st.sidebar.radio('By : Irfadillah Afni Nurvita', ['Tentang saya', 
                                                 'Proyek', 'Prediksi', 
                                                 'Kontak'])

if page == 'Tentang saya':
    import about_me
    about_me.about_me()
elif page == 'Proyek':
    import project
    project.project()
elif page == 'Prediksi':
    import prediksi
    prediksi.prediction()
elif page == 'Kontak':
    import kontak
    kontak.kontak()

