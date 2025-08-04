import streamlit as st

st.set_page_config(layout='wide')
col1, col2 = st.columns([4, 1]) 
with col1:
    st.title('Analisis Fasilitas & Lokasi dengan Machine Learning')
with col2:
    st.image('https://www.bukitvista.com/wp-content/uploads/2021/06/BukitVista-LOGO-ONLY-transparent.png', width=120)
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
