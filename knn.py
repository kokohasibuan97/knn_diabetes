import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="KNN Risiko Diabetes", layout="centered")
st.title("Prediksi Risiko Diabetes (KNN)")

# Dataset 10 pasien
data = pd.DataFrame({
    "Pasien": ["P01","P02","P03","P04","P05","P06","P07","P08","P09","P10"],
    "Usia": [25,30,35,40,45,50,55,28,38,48],
    "BMI": [21.5,23.1,25.0,27.5,30.2,31.5,33.0,22.5,26.8,29.5],
    "Glukosa": [90,95,105,120,140,155,165,92,115,145],
    "Tekanan Darah": [110,115,120,125,135,140,145,112,122,130],
    "Riwayat Keluarga": [0,0,0,1,1,1,1,0,0,1],
    "Kelas": ["Rendah","Rendah","Rendah","Tinggi","Tinggi",
              "Tinggi","Tinggi","Rendah","Rendah","Tinggi"],
})
fitur = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]

st.subheader("Dataset")
st.dataframe(data, hide_index=True)

st.subheader("Input Pasien Baru")
c1, c2 = st.columns(2)
usia = c1.number_input("Usia", 1, 120, 42)
bmi = c2.number_input("BMI", 10.0, 60.0, 28.5, step=0.1)
glukosa = c1.number_input("Glukosa", 50, 400, 130)
td = c2.number_input("Tekanan Darah", 50, 250, 128)
riwayat = st.selectbox("Riwayat Keluarga", [0, 1],
                       format_func=lambda x: "Ada" if x == 1 else "Tidak ada",
                       index=1)
k = st.slider("Nilai K", 1, 9, 3, step=2)

if st.button("Prediksi"):
    baru = np.array([usia, bmi, glukosa, td, riwayat], dtype=float)
    X = data[fitur].values.astype(float)

    # Jarak Euclidean
    data["Jarak"] = np.sqrt(((X - baru) ** 2).sum(axis=1))
    terdekat = data.sort_values("Jarak").head(k)

    st.subheader("Jarak ke Setiap Pasien")
    st.dataframe(data.sort_values("Jarak")[["Pasien", "Kelas", "Jarak"]],
                 hide_index=True)

    st.subheader(f"{k} Tetangga Terdekat")
    st.dataframe(terdekat[["Pasien", "Kelas", "Jarak"]], hide_index=True)

    # Voting mayoritas
    vote = terdekat["Kelas"].value_counts()
    hasil = vote.idxmax()
    st.write("Hasil voting:", vote.to_dict())
    if hasil == "Tinggi":
        st.error(f"Prediksi: Risiko **TINGGI** (kelas 1)")
    else:
        st.success(f"Prediksi: Risiko **RENDAH** (kelas 0)")