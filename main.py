import streamlit as st

st.set_page_config(layout='wide')

st.markdown(
    """
    <style>
    .logo-container {
        text-align: center;
        margin-bottom: 10px;
    }
    .judul-container {
        text-align: center;
        font-size: 3.5rem;  /* Ukuran font lebih besar */
        font-weight: bold;
        margin-top: 0;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="logo-container">
        <img src="https://www.bukitvista.com/wp-content/uploads/2021/06/BukitVista-LOGO-ONLY-transparent.png" width="300" />
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="judul-container">
        Analisis Fasilitas & Lokasi dengan Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    '',
    ['Tentang saya', 'Proyek', 'Prediksi', 'Kontak'],
    horizontal=True
)

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
