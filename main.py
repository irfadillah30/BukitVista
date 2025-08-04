import streamlit as st

st.set_page_config(layout='wide')
st.markdown(
    """
    <style>
    .container-flex {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .container-flex h1 {
        font-size: 2rem;  /* kecilkan font judul supaya nggak terlalu besar */
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="container-flex">
        <h1>Analisis Fasilitas & Lokasi dengan Machine Learning</h1>
        <img src="https://www.bukitvista.com/wp-content/uploads/2021/06/BukitVista-LOGO-ONLY-transparent.png" width="100" />
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.radio(
    ' ',
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
