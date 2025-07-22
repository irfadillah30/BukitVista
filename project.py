import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import extract_facilities


def project():
    st.title("Tentang Bukit Vista")
    st.write("""
    Bukit Vista adalah perusahaan yang bergerak di bidang penyewaan vila dan properti di Bali.
    Perusahaan tersebut menyediakan berbagai pilihan vila yang dilengkapi dengan fasilitas lengkap untuk memastikan pengalaman menginap yang nyaman bagi para tamu.
    """)

    st.title("Tujuan Proyek Ini")
    st.write("""
    Selamat datang di halaman proyek!
    Tujuan dari proyek ini adalah:
    - Menentukan harga sewa properti berdasarkan fasilitas yang tersedia.
    - Membangun model prediksi harga sewa vila.
    """)

    st.title("Data")
    df = pd.read_csv("Daftar_vila_Dibali_Bukit_Vista_Cleaned.csv")

    # Format kolom Harga
    df['Harga'] = df['Harga'].astype(str)
    df['Harga'] = (
        df['Harga']
        .str.replace('\u00a5', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.strip()
    )
    df['Harga'] = df['Harga'].replace('', np.nan)
    df['Harga'] = df['Harga'].astype(float)
    df['Harga'] = df['Harga'].fillna(df['Harga'].median()).astype(int)

    # Konversi kolom Minimum menginap
    df['Minimum menginap'] = pd.to_numeric(df['Minimum menginap'], errors='coerce')

    # Ekstraksi Fasilitas dengan pengecekan panjang list
    facilities = df['Fasilitas'].apply(extract_facilities)
    df['Jumlah Tamu'] = facilities.apply(lambda x: x[0] if x and len(x) > 0 else 0)
    df['Jumlah Kamar Tidur'] = facilities.apply(lambda x: x[1] if x and len(x) > 1 else 0)
    df['Jumlah Tempat Tidur'] = facilities.apply(lambda x: x[2] if x and len(x) > 2 else 0)
    df['Jumlah Kamar Mandi'] = facilities.apply(lambda x: x[3] if x and len(x) > 3 else 0)

    st.write("""
    Data yang digunakan mencakup:
    - Harga sewa
    - Fasilitas (jumlah tamu, kamar, tempat tidur, kamar mandi)
    - Lokasi
    - Rating dan durasi menginap

    Sumber: [Bukit Vista](https://www.bukitvista.com/bali-long-term-rentals)
    """)

    st.title("Visualisasi Exploratory Data Analysis")

    # Distribusi Harga
    st.header("Distribusi Harga Vila")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Harga'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribusi Harga Vila')
    ax.set_xlabel('Harga (¥)')
    ax.set_ylabel('Frekuensi')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Harga per Lokasi
    st.header("Distribusi Harga Berdasarkan Lokasi")
    order = df.groupby('Source_Location')['Harga'].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Source_Location', y='Harga', data=df, order=order, palette='viridis', ax=ax)
    ax.set_title('Distribusi Harga per Lokasi')
    ax.set_xlabel('Lokasi')
    ax.set_ylabel('Harga (¥)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Jumlah Tamu vs Harga
    st.header("Jumlah Tamu vs Harga")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(x='Jumlah Tamu', y='Harga', data=df, color='royalblue', alpha=0.5, s=100, edgecolor='gray', ax=ax)
    sns.regplot(x='Jumlah Tamu', y='Harga', data=df, scatter=False, color='darkred', line_kws={'linewidth': 2}, ax=ax)
    ax.set_title('Jumlah Tamu vs Harga')
    ax.set_xlabel('Jumlah Tamu')
    ax.set_ylabel('Harga (¥)')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Jumlah Kamar Tidur vs Harga
    st.header("Jumlah Kamar Tidur vs Harga")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(x='Jumlah Kamar Tidur', y='Harga', data=df, color='royalblue', alpha=0.5, s=100, edgecolor='gray', ax=ax)
    sns.regplot(x='Jumlah Kamar Tidur', y='Harga', data=df, scatter=False, color='darkred', line_kws={'linewidth': 2}, ax=ax)
    ax.set_title('Jumlah Kamar Tidur vs Harga')
    ax.set_xlabel('Jumlah Kamar Tidur')
    ax.set_ylabel('Harga (¥)')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Jumlah Tempat Tidur vs Harga
    st.header("Jumlah Tempat Tidur vs Harga")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(x='Jumlah Tempat Tidur', y='Harga', data=df, color='royalblue', alpha=0.5, s=100, edgecolor='gray', ax=ax)
    sns.regplot(x='Jumlah Tempat Tidur', y='Harga', data=df, scatter=False, color='darkred', line_kws={'linewidth': 2}, ax=ax)
    ax.set_title('Jumlah Tempat Tidur vs Harga')
    ax.set_xlabel('Jumlah Tempat Tidur')
    ax.set_ylabel('Harga (¥)')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Korelasi Fitur
    st.header("Korelasi Fitur Numerik")
    corr = df[['Jumlah Tamu', 'Jumlah Kamar Tidur', 'Jumlah Tempat Tidur', 'Harga']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Matriks Korelasi Fitur Numerik')
    st.pyplot(fig)
    st.dataframe(corr)
