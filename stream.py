import tensorflow as tf
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the Bi-LSTM model
model_bilstm = load_model('model_bilstm14.h5')

# Load tokenizer
with open('tokenizer14.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Judul halaman
st.title('Analisis Kalimat AI dan Manusia')
st.write("""
Aplikasi ini memprediksi apakah sebuah kalimat adalah buatan manusia atau AI berdasarkan model Bi-LSTM.
""")

# Input teks
input_text = st.text_area('Masukkan Kalimat untuk Analisis', height=150)

# Placeholder hasil prediksi
prediction_result = ''

if st.button('Lakukan Analisis'):
    if input_text.strip():
        # Preprocessing input
        sequence = tokenizer.texts_to_sequences([input_text])
        padded_sequence = pad_sequences(sequence, maxlen=100)  # Sesuaikan dengan panjang input model Anda

        # Prediksi
        prediction = model_bilstm.predict(padded_sequence)
        label = np.round(prediction).astype(int)  # Pembulatan ke 0 atau 1

        if label == 0:
            prediction_result = 'Kalimat ini adalah buatan manusia.'
        elif label == 1:
            prediction_result = 'Kalimat ini adalah buatan AI.'
        else:
            prediction_result = 'Tidak dapat mengidentifikasi kalimat.'

        st.success(prediction_result)
    else:
        st.warning('Silakan masukkan kalimat untuk dianalisis.')

# Footer
st.write("Model menggunakan pendekatan Bi-LSTM dengan Word2Vec untuk representasi fitur.")
