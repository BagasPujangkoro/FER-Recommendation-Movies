import cv2
from deepface import DeepFace
import numpy as np
import streamlit as st

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Uji Akurasi FER - Bagas Pujangkoro",
    page_icon="🎥",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #141414; color: #ffffff; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .main-header { font-size: 2.2rem; font-weight: 800; color: #E50914; text-align: center; margin-bottom: 0px; }
    .sub-header { font-size: 1rem; color: #b3b3b3; text-align: center; margin-bottom: 30px; }
    .card { background-color: #1f1f1f; border: 1px solid #333333; padding: 20px; border-radius: 10px; margin-top: 20px; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-header">Evaluasi Responden: Facial Emotion Recognition (FER)</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">BAGAS PUJANGKORO Program Studi Teknologi Informasi - Universitas Gadjah Mada</p>',
    unsafe_allow_html=True,
)
st.divider()

st.markdown("### 📸 Ambil Foto Wajah Anda")
st.markdown(
    "Sistem ini dirancang untuk mendeteksi emosi wajah secara otomatis (dilengkapi penstabil cahaya *CLAHE* untuk lingkungan minim cahaya)."
)

# Tangkap input kamera dari responden
camera_image = st.camera_input("Silahkan ambil foto wajah Anda")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Balikkan gambar secara horizontal (1) agar tidak mirror / tidak terbalik
    frame = cv2.flip(frame, 1)

    with st.spinner("Menganalisis ekspresi wajah responden..."):
        try:
            # 1. Cek tingkat kecerahan (Grayscale)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_frame)

            # 2. Terapkan CLAHE jika pencahayaan rendah (< 70)
            if avg_brightness < 70:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                st.toast(
                    "⚠️ Pencahayaan rendah terdeteksi. Auto-Enhancement (CLAHE) diterapkan!",
                    icon="💡",
                )

            # 3. Analisis Emosi dengan DeepFace
            analysis = DeepFace.analyze(
                frame, actions=["emotion"], enforce_detection=False
            )
            dom_emotion = str(analysis[0]["dominant_emotion"]).capitalize()
            emotions_score = analysis[0]["emotion"]

            # Tampilkan Hasil ke Responden
            st.success("Analisis Berhasil!")

            st.markdown(
                f"""
                <div class="card">
                    <h3 style="color: #E50914; margin-top:0;">Hasil Deteksi Emosi:</h3>
                    <p style="font-size: 1.3rem; font-weight: bold;">Emosi Dominan: {dom_emotion}</p>
                    <p style="color: #aaa; font-size: 0.9rem;">Tingkat Keterangan Gambar: {int(avg_brightness)}/255</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tampilkan rincian probabilitas emosi lainnya sebagai bentuk validasi ilmiah
            with st.expander("Lihat Detail Probabilitas Semua Emosi"):
                st.json(emotions_score)

        except Exception as e:
            st.error(
                f"Gagal mendeteksi wajah. Pastikan wajah terlihat jelas di kamera. Detail: {e}"
            )