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

    # Bersihkan dan ubah kolom Harga ke numeric
    df['Harga'] = df['Harga'].astype(str)
    df['Harga'] = df['Harga'].str.replace('¥', '', regex=False).str.replace('.', '', regex=False).str.strip()
    df['Harga'] = df['Harga'].replace('', np.nan).astype(float)
    df['Harga'] = df['Harga'].fillna(df['Harga'].median()).astype(int)

    df['Minimum menginap'] = pd.to_numeric(df['Minimum menginap'], errors='coerce')

    facilities = df['Fasilitas'].apply(extract_facilities)
    df['Jumlah Tamu'] = facilities.apply(lambda x: x[0])
    df['Jumlah Kamar Tidur'] = facilities.apply(lambda x: x[1])
    df['Jumlah Tempat Tidur'] = facilities.apply(lambda x: x[2])

    # Ganti 0 dengan 1 agar slider tidak error
    df['Jumlah Tamu'] = df['Jumlah Tamu'].replace(0, 1)
    df['Jumlah Kamar Tidur'] = df['Jumlah Kamar Tidur'].replace(0, 1)

    # Nilai min dan max slider
    tamu_min, tamu_max = int(df['Jumlah Tamu'].min()), int(df['Jumlah Tamu'].max())
    kamar_min, kamar_max = int(df['Jumlah Kamar Tidur'].min()), int(df['Jumlah Kamar Tidur'].max())

    # visualisasi
    st.markdown("## 📈 Visualisasi Data 📉") 
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(" 🛏️ Tempat Tidur vs Harga")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x='Jumlah Tempat Tidur', y='Harga', data=df, color='teal', alpha=0.6, ax=ax1)
        sns.regplot(x='Jumlah Tempat Tidur', y='Harga', data=df, scatter=False, color='red', ax=ax1)
        ax1.set_title('Jumlah Tempat Tidur vs Harga')
        st.pyplot(fig1)

    with col2:
        st.markdown(" 🛏️ Kamar Tidur vs Harga")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x='Jumlah Kamar Tidur', y='Harga', data=df, color='purple', alpha=0.6, ax=ax2)
        sns.regplot(x='Jumlah Kamar Tidur', y='Harga', data=df, scatter=False, color='red', ax=ax2)
        ax2.set_title('Jumlah Kamar Tidur vs Harga')
        st.pyplot(fig2)
        
    with col3:
        st.markdown(" 👤 Jumlah Tamu vs Harga")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x='Jumlah Tamu', y='Harga', data=df, color='orange', alpha=0.6, ax=ax3)
        sns.regplot(x='Jumlah Tamu', y='Harga', data=df, scatter=False, color='red', ax=ax3)
        ax3.set_title('Jumlah Tamu vs Harga')
        st.pyplot(fig3)

    #   DASHBOARD INTERAKTIF 
    st.markdown("## ⏳ Filter Vila")

    col1, col2, col3 = st.columns(3)  # Sekarang hanya 3 kolom
    lokasi_options = df['Source_Location'].unique()

    with col1:
        selected_lokasi = st.selectbox("📍 Lokasi", ['Semua'] + sorted(lokasi_options.tolist()))
    with col2:
        selected_kamar = st.slider("🛌 Jumlah Kamar Tidur", kamar_min, kamar_max, (kamar_min, kamar_max))
    with col3:
        selected_tamu = st.slider("👤 Jumlah Tamu", tamu_min, tamu_max, (tamu_min, tamu_max))

    # Filter Data
    df_filtered = df.copy()
    if selected_lokasi != 'Semua':
        df_filtered = df_filtered[df_filtered['Source_Location'] == selected_lokasi]

    df_filtered = df_filtered[
        (df_filtered['Jumlah Kamar Tidur'] >= selected_kamar[0]) &
        (df_filtered['Jumlah Kamar Tidur'] <= selected_kamar[1]) &
        (df_filtered['Jumlah Tamu'] >= selected_tamu[0]) &
        (df_filtered['Jumlah Tamu'] <= selected_tamu[1])
    ]
    st.markdown(f"#### Menampilkan {len(df_filtered)} vila berdasarkan filter")

    if len(df_filtered) == 0:
        st.warning("Maaf, tidak ada vila yang sesuai dengan filter. Silakan coba ubah jumlah tamu, kamar, atau lokasi.")
    else:
        st.subheader("📍 Jumlah Vila per Lokasi")
        jumlah_vila = df_filtered['Source_Location'].value_counts().sort_values(ascending=True)

        fig4, ax4 = plt.subplots(figsize=(8, 6))
        jumlah_vila.plot(kind='barh', color='cornflowerblue', ax=ax4)

        # angka di setiap bar
        for i, v in enumerate(jumlah_vila):
            ax4.text(v + 0.1, i, str(v), color='blue', va='center', fontweight='bold')

        ax4.set_xlabel('Jumlah Vila')
        ax4.set_title('Jumlah Vila per Lokasi (Setelah Filter)')
        ax4.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig4)




    st.markdown("---")
    st.markdown("by : Irfadillah Afni Nurvita")