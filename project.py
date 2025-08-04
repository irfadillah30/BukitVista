import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import extract_facilities

def project():
    st.title("📊 Dashboard Vila Bukit Vista")

    with st.expander("Tentang Bukit Vista"):
        st.write("""Bukit Vista adalah perusahaan yang bergerak di bidang penyewaan vila dan properti di Bali.
    Perusahaan tersebut menyediakan berbagai pilihan vila yang dilengkapi dengan fasilitas lengkap untuk memastikan pengalaman menginap yang nyaman bagi para tamu.
    """)

    with st.expander("Tujuan Proyek"):
        st.markdown("""
        - Menentukan harga sewa properti berdasarkan fasilitas yang tersedia
        - Membangun model prediksi harga sewa vila
        """)

    df = pd.read_csv("Daftar_vila_Dibali_Bukit_Vista_Cleaned.csv")

    df['Harga'] = df['Harga'].astype(str)
    df['Harga'] = df['Harga'].str.replace('¥', '', regex=False).str.replace('.', '', regex=False).str.strip()
    df['Harga'] = df['Harga'].replace('', np.nan).astype(float)
    df['Harga'] = df['Harga'].fillna(df['Harga'].median()).astype(int)
    df['Minimum menginap'] = pd.to_numeric(df['Minimum menginap'], errors='coerce')

    facilities = df['Fasilitas'].apply(extract_facilities)
    df['Jumlah Tamu'] = facilities.apply(lambda x: x[0])
    df['Jumlah Kamar Tidur'] = facilities.apply(lambda x: x[1])
    df['Jumlah Tempat Tidur'] = facilities.apply(lambda x: x[2])

    # Filter lokasi
    st.subheader("Filter Data  ")
    lokasi_terpilih = st.multiselect(
        "Pilih Lokasi", options=df['Source_Location'].unique(), default=list(df['Source_Location'].unique())
    )

    # Metric Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata Harga", f"¥{int(df['Harga'].mean()):,}".replace(',', '.'))
    col2.metric("Jumlah Vila", len(df))
    col3.metric("Rata-rata Tamu", round(df['Jumlah Tamu'].mean(), 1))

    st.markdown("---")

    # Visualisasi Harga
    st.subheader("📈 Distribusi Harga Vila")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Harga'], bins=30, kde=True, color='skyblue', ax=ax1)
    ax1.set_title('Distribusi Harga Vila')
    st.pyplot(fig1)

    # Harga per Lokasi
    st.subheader("📍 Harga Berdasarkan Lokasi")
    order = df.groupby('Source_Location')['Harga'].mean().sort_values().index
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Source_Location', y='Harga', data=df, order=order, palette='viridis', ax=ax2)
    ax2.set_title('Harga per Lokasi')
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # Scatterplots Fitur
    fitur_visual = {
        "👥 Jumlah Tamu": 'Jumlah Tamu',
        "🛏️ Kamar Tidur": 'Jumlah Kamar Tidur',
        "🛌 Tempat Tidur": 'Jumlah Tempat Tidur'
    }

    for title, col in fitur_visual.items():
        st.subheader(f"{title} vs Harga")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x=col, y='Harga', data=df, alpha=0.6, s=100, edgecolor='gray', ax=ax)
        sns.regplot(x=col, y='Harga', data=df, scatter=False, color='red', ax=ax)
        st.pyplot(fig)

    st.caption("Sumber data: Bukit Vista | Disusun oleh Irfadillah Afni Nurvita")
