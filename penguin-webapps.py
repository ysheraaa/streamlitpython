import streamlit as st 
import pandas as pd
import numpy as np 
from PIL import Image
import pickle 
from sklearn.naive_bayes import GaussianNB

st.write("""
# Klasifikasi Penguin (Web Apps) 
Aplikasi berbasis web untuk memprediksi (mengklasifikasi) jenis penguin **Palmer Penguin**. \n
Data yang didapat dari [palmerpenguin library] (https://github.com/allisonhorst/palmerpenguins)    
dalam bentuk R oleh Allison Horst.    
""")

img = Image.open('penguin.png')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

img2 = Image.open('paruh.png')
img2 = img2.resize((700, 418))
st.image(img2, use_column_width=False)

st.sidebar.header('Parameter Inputan')

#Upload File CSV untuk Parameter Inputan

upload_file = st.sidebar.file_uploader("Upload file CSV Anda", type=["csv"])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        pulau = st.sidebar.selectbox('Pulau', ('Biscoe','Dream','Torgersen'))
        gender = st.sidebar.selectbox('Gender', ('pria', 'wanita'))
        panjang_paruh = st.sidebar.slider('Panjang Paruh (mm)', 32.1, 59.6,43.9)
        kedalaman_paruh = st.sidebar.slider('Kedalaman Paruh (mm)', 13.1, 21.5,17.2)
        panjang_sirip = st.sidebar.slider('Panjang Sirip (mm)', 172.0,231.0,201.0)
        masa_tubuh = st.sidebar.slider('Masa Tubuh (g)', 2700.0, 6300.0, 4207.0)
        data = {'pulau' : pulau,
                'panjang_paruh_mm' : panjang_paruh,
                'kedalaman_paruh_mm' : kedalaman_paruh,
                'panjang_sirip_mm' : panjang_sirip,
                'masa_tubuh_g' : masa_tubuh,
                'gender' : gender}
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()
  
#menggabungkan inputan dan dataset penguin  
penguin_raw = pd.read_csv('penguin.csv')
penguin = penguin_raw.drop(columns=['jenis'])
df = pd.concat([inputan, penguin], axis = 0)

#encode untuk fitur ordinal number / dibuat kedalam numerik
encode = ['gender', 'pulau']
for col in encode: 
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
    
df = df[:1] #ambil baris pertama (input data user)

#menampilkan parameter inputan 
st.subheader('Parameter Inputan')

if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file csv untuk diupload. Saat ini memakai sampel inputan(seperti tampilan dibawah)')    
    st.write(df)
    
#Load model NBC naive bayes classification
load_model = pickle.load(open('modelNBC_penguin.pkl', 'rb'))

#terapkan naive bayes classification
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
jenis_penguin = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(jenis_penguin)

st.subheader('Hasil Prediksi (Klasifikasi Penguin)')
st.write(jenis_penguin[prediksi])

st.subheader('Probabilitas Hasil Prediksi (Klasifikasi Penguin)')
st.write(prediksi_proba)
