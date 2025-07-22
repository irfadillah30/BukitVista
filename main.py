import streamlit as st

st.set_page_config(layout='wide')
st.title('Analisis Fasilitas & Lokasi dengan Machine Learning')

st.sidebar.image('https://www.bukitvista.com/wp-content/uploads/2021/06/BukitVista-LOGO-ONLY-transparent.png', width=200)
st.header('Project Data Scientist')

st.sidebar.title('Model Prediksi Harga Vila')
page = st.sidebar.radio('By : Irfadillah Afni Nurvita', ['Tentang saya', 'Proyek', 'Prediksi', 'Kontak'])

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
