import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="KNN Risiko Diabetes", layout="centered")

# ================= HEADER =================
st.title("Prediksi Risiko Diabetes dengan KNN")
# Ganti titik-titik berikut dengan identitas kamu
st.caption("Nama: Koko Bonardo Hasibuan | NIM: 2402184 | Mata Kuliah: Data Mining")

# ================= PENJELASAN =================
with st.expander("Penjelasan Aplikasi", expanded=True):
    st.markdown(
        """
Aplikasi ini membantu klinik mengidentifikasi apakah seorang pasien memiliki
**risiko diabetes tinggi atau rendah** menggunakan algoritma
**K-Nearest Neighbors (KNN)**.

**Atribut yang digunakan:** Usia, BMI, Glukosa, Tekanan Darah, dan
Riwayat Diabetes dalam Keluarga.

**Kelas target:** `0` = Risiko Rendah, `1` = Risiko Tinggi.

**Langkah KNN:**
1. Hitung jarak pasien baru ke seluruh data latih.
2. Urutkan dari jarak terkecil.
3. Ambil K tetangga terdekat.
4. Tentukan kelas dengan voting mayoritas.
"""
    )
    st.markdown("**Rumus jarak Euclidean:**")
    st.latex(r"d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}")

# ================= DATASET =================
data = pd.DataFrame({
    "Pasien": ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"],
    "Usia": [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
    "BMI": [21.5, 23.1, 25.0, 27.5, 30.2, 31.5, 33.0, 22.5, 26.8, 29.5],
    "Glukosa": [90, 95, 105, 120, 140, 155, 165, 92, 115, 145],
    "Tekanan Darah": [110, 115, 120, 125, 135, 140, 145, 112, 122, 130],
    "Riwayat Keluarga": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    "Kelas": ["Rendah", "Rendah", "Rendah", "Tinggi", "Tinggi",
              "Tinggi", "Tinggi", "Rendah", "Rendah", "Tinggi"],
})
fitur = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]

st.subheader("Dataset Pasien")
st.dataframe(data, hide_index=True)

# ================= INPUT =================
st.subheader("Input Pasien Baru")
c1, c2 = st.columns(2)
usia = c1.number_input("Usia", 1, 120, 42)
bmi = c2.number_input("BMI", 10.0, 60.0, 28.5, step=0.1)
glukosa = c1.number_input("Glukosa", 50, 400, 130)
td = c2.number_input("Tekanan Darah", 50, 250, 128)
riwayat = st.selectbox(
    "Riwayat Keluarga",
    [0, 1],
    format_func=lambda x: "Ada (1)" if x == 1 else "Tidak ada (0)",
    index=1,
)

st.subheader("Pengaturan")
k = st.slider("Nilai K", 1, 9, 3, step=2)
normalisasi = st.checkbox(
    "Gunakan normalisasi Min-Max",
    value=False,
    help="Menyamakan skala semua atribut ke rentang 0-1 agar atribut "
         "berskala besar (misalnya Glukosa) tidak terlalu mendominasi jarak.",
)

# ================= PREDIKSI =================
if st.button("Prediksi"):
    baru = np.array([usia, bmi, glukosa, td, riwayat], dtype=float)
    X = data[fitur].values.astype(float)

    if normalisasi:
        mn, mx = X.min(axis=0), X.max(axis=0)
        Xp = (X - mn) / (mx - mn)
        bp = (baru - mn) / (mx - mn)

        st.subheader("Data Setelah Normalisasi Min-Max")
        st.latex(r"x' = \frac{x - x_{min}}{x_{max} - x_{min}}")
        norm_df = pd.DataFrame(Xp, columns=fitur).round(4)
        norm_df.insert(0, "Pasien", data["Pasien"])
        baru_df = pd.DataFrame([bp], columns=fitur).round(4)
        baru_df.insert(0, "Pasien", "Baru")
        st.dataframe(pd.concat([norm_df, baru_df], ignore_index=True),
                     hide_index=True)
    else:
        Xp, bp = X, baru

    # Jarak Euclidean
    hasil = data[["Pasien", "Kelas"]].copy()
    hasil["Jarak"] = np.sqrt(((Xp - bp) ** 2).sum(axis=1)).round(4)
    urut = hasil.sort_values("Jarak").reset_index(drop=True)
    urut.insert(0, "Peringkat", range(1, len(urut) + 1))

    st.subheader("Jarak ke Setiap Pasien (Terurut)")
    st.dataframe(urut, hide_index=True)

    terdekat = urut.head(k)
    st.subheader(f"{k} Tetangga Terdekat")
    st.dataframe(terdekat, hide_index=True)

    # Voting mayoritas
    vote = terdekat["Kelas"].value_counts()
    hasil_kelas = vote.idxmax()
    st.write("Hasil voting:", vote.to_dict())

    if hasil_kelas == "Tinggi":
        st.error("Prediksi: Risiko **TINGGI** (kelas 1)")
    else:
        st.success("Prediksi: Risiko **RENDAH** (kelas 0)")

st.divider()
st.caption("Dibuat dengan Python dan Streamlit | Algoritma: K-Nearest Neighbors")