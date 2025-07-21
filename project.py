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



def project():
    st.title("Tentang Bukit Vista")
    st.write("""
    Bukit Vista adalah perusahaan yang bergerak di bidang penyewaan vila dan properti di Bali.
    Perusahaan tersebut menyediakan berbagai pilihan vila yang dilengkapi dengan fasilitas lengkap untuk memastikan pengalaman menginap yang nyaman bagi para tamu.
    """)
    
    st.title("Tujuan proyek ini")
    st.write("""
    Selamat datang di halaman proyek !!!
    Di sini, saya telah mengerjakan beberapa project sebagai seorang data scientist.
    Adapun tujuan dari project yang dibuat ini adalah: 
    - Menentukan harga sewa properti sesuai dengan fasilitas yang ada.
    - Membangun model prediksi.
    
    """)
    
    st.title("Data")
    st.write("""
    Data yang digunakan dalam proyek ini adalah data vila yang mencakup berbagai fitur seperti harga,
    fasilitas, lokasi, dan lainnya. Data ini diambil dari sumber yang terpercaya dan telah dibersihkan
    untuk memastikan kualitas analisis.
    """)
    st.write("""
             Web : [Bukit Vista](https://www.bukitvista.com/bali-long-term-rentals)
             
             Data Properti: 
             
               - Harga sewa
               - Fasilitas
               - Lokasi

            Data Pelanggan:
               - Rating
               - Durasi menginap.
    """)
    df = pd.read_csv('Daftar_vila_Dibali_Bukit_Vista_Cleaned.csv')

      # Ubah tipe harga
      # Menghapus simbol "¥" dan titik, lalu konversi ke angka
    df['Harga'] = df['Harga'].astype(str)
    df['Harga'] = df['Harga'].str.replace('¥', '', regex=False).str.replace('.', '', regex=False).str.strip()

      # Isi NaN dengan median sebelum konversi ke int
    df['Harga'] = df['Harga'].replace('', np.nan)
    df['Harga'] = df['Harga'].astype(float)
    median_harga = df['Harga'].median()
    df['Harga'] = df['Harga'].fillna(median_harga).astype(int)

      # Ubah 'Minimum menginap' ke numerik
    df['Minimum menginap'] = pd.to_numeric(df['Minimum menginap'], errors='coerce')


      # Ekstrak fitur dari kolom Fasilitas 
    df['Jumlah_Tamu'] = df['Fasilitas'].str.extract(r'(\d+)\s*tamu').astype(float)
    df['Jumlah_Kamar'] = df['Fasilitas'].str.extract(r'(\d+)\s*kamar tidur').astype(float)
    df['Jumlah_Tempat_Tidur'] = df['Fasilitas'].str.extract(r'(\d+)\s*tempat tidur').astype(float)
    df['Jumlah_Kamar_Mandi'] = df['Fasilitas'].str.extract(r'(\d+)\s*kamar mandi').astype(float)  
   
   
   
   
    st.title("Visualisasi Exploratori Data Analisis")
    st.header("Visualisasi Distribusi harga")
    st.write("""    Di halaman ini, saya telah membuat visualisasi distribusi harga sewa vila untuk memberikan gambaran
    tentang bagaimana harga sewa bervariasi berdasarkan berbagai fitur seperti fasilitas dan lokasi.
    Visualisasi ini membantu dalam memahami pola harga dan memberikan wawasan yang berguna untuk pengambilan keputusan.
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Harga'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribusi Harga Vila', fontsize=16)
    ax.set_xlabel('Harga (¥)', fontsize=12)
    ax.set_ylabel('Frekuensi', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    
    st.header("Visualisasi Distribusi Harga Berdasarkan Lokasi")
    st.write("""    Di halaman ini, saya telah membuat visualisasi distribusi harga sewa vila berdasarkan lokasi.
    Visualisasi ini membantu dalam memahami bagaimana harga sewa bervariasi di berbagai lokasi di Bali.
    Dengan informasi ini, kita dapat mengidentifikasi lokasi-lokasi yang memiliki harga sewa yang lebih tinggi atau lebih rendah,
    serta memahami faktor-faktor yang mempengaruhi harga sewa vila di setiap lokasi.
    """)
    # Urutkan lokasi berdasarkan rata-rata harga
    order = df.groupby('Source_Location')['Harga'].mean().sort_values().index

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
         x='Source_Location',
         y='Harga',
         data=df,
         order=order,
         palette='viridis',
         ax=ax
      )

      # Atur tampilan
    ax.set_title('Distribusi Harga per Lokasi', fontsize=16)
    ax.set_xlabel('Lokasi', fontsize=12)
    ax.set_ylabel('Harga (¥)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    
    st.header("Visualisasi Distribusi Jumlah Tamu vs Harga")
    st.write("""    Di halaman ini, saya telah membuat visualisasi distribusi jumlah tamu yang menginap di vila. 
      Visualisasi ini membantu dalam memahami bagaimana jumlah tamu mempengaruhi harga sewa vila.
      Dengan informasi ini, kita dapat mengidentifikasi pola-pola tertentu dalam harga sewa berdasarkan jumlah tamu yang menginap.
      """)
    sns.set(style='whitegrid', palette='pastel')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
         x='Jumlah_Tamu',
         y='Harga',
         data=df,
         color='royalblue',
         alpha=0.5,
         s=100,
         edgecolor='gray',
         ax=ax
      )
    sns.regplot(
         x='Jumlah_Tamu',
         y='Harga',
         data=df,
         scatter=False,
         color='darkred',
         line_kws={'linewidth': 2},
         ax=ax
      )
    ax.set_title('Hubungan Jumlah Tamu dan Harga Sewa Vila', fontsize=16, fontweight='bold')
    ax.set_xlabel('Jumlah Tamu', fontsize=12)
    ax.set_ylabel('Harga (¥)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.header("Visualisasi Jumlah Kamar Tidur vs Harga")
    st.write("""    Di halaman ini, saya telah membuat visualisasi jumlah kamar tidur yang tersedia di vila. 
         Visualisasi ini membantu dalam memahami bagaimana jumlah kamar tidur mempengaruhi harga sewa vila.
         Dengan informasi ini, kita dapat mengidentifikasi pola-pola tertentu dalam harga sewa berdasarkan jumlah kamar tidur yang tersedia.
         """)
    sns.set(style='whitegrid', palette='pastel')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
         x='Jumlah_Kamar',
         y='Harga',
         data=df,
         color='royalblue',
         alpha=0.5,
         s=100,
         edgecolor='gray',
         ax=ax
      )
    sns.regplot(
         x='Jumlah_Kamar',
         y='Harga',
         data=df,
         scatter=False,
         color='darkred',
         line_kws={'linewidth': 2},
         ax=ax
      )
    ax.set_title('Hubungan Jumlah Kamar Tidur dan Harga Sewa Vila', fontsize=16, fontweight='bold')
    ax.set_xlabel('Jumlah Kamar Tidur', fontsize=12)
    ax.set_ylabel('Harga (¥)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)
    
    
    st.header("Visualisasi Jumlah Tempat Tidur vs Harga")
    st.write("""    Di halaman ini, saya telah membuat visualisasi jumlah tempat tidur yang tersedia di vila. 
            Visualisasi ini membantu dalam memahami bagaimana jumlah tempat tidur mempengaruhi harga sewa vila.
            Dengan informasi ini, kita dapat mengidentifikasi pola-pola tertentu dalam harga sewa berdasarkan jumlah tempat tidur yang tersedia.
            """)
    sns.set(style='whitegrid', palette='pastel')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
           x='Jumlah_Tempat_Tidur',
           y='Harga',
           data=df,
           color='royalblue',
           alpha=0.5,
           s=100,
           edgecolor='gray',
           ax=ax
        )
    sns.regplot(
           x='Jumlah_Tempat_Tidur',
           y='Harga',
           data=df,
           scatter=False,
           color='darkred',
           line_kws={'linewidth': 2},
           ax=ax
        )
    ax.set_title('Hubungan Jumlah Tempat Tidur dan Harga Sewa Vila', fontsize=16, fontweight='bold')
    ax.set_xlabel('Jumlah Tempat Tidur', fontsize=12)
    ax.set_ylabel('Harga (¥)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.header("Korelasi Fitur")
    st.write("""    Di halaman ini, saya telah membuat visualisasi korelasi antara fitur-fitur yang ada dalam dataset.
      Visualisasi ini membantu dalam memahami hubungan antara berbagai fitur dan harga sewa vila.
      Dengan informasi ini, kita dapat mengidentifikasi fitur-fitur yang memiliki pengaruh signifikan
      terhadap harga sewa vila, serta memahami pola-pola yang mungkin ada dalam data.
      """)

    corr = df[['Jumlah_Tamu', 'Jumlah_Kamar', 'Jumlah_Tempat_Tidur', 'Harga']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Matriks Korelasi Fitur Numerik', fontsize=16)
    st.pyplot(fig)
    st.write("Korelasi antara Harga, Jumlah Tamu, Jumlah Tempat Tidur, Jumlah Kamar:")
    st.dataframe(corr)
      
