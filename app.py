import cv2
import re
from deepface import DeepFace
from deep_translator import GoogleTranslator
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Cinemagic AI - Mood-Aware PUSBPG Recommender",
    page_icon="🎬",
    layout="wide",
)

# Custom CSS Ala Platform Streaming (Netflix / IMDb Dark Theme)
st.markdown(
    """
    <style>
    .stApp { background-color: #141414; color: #ffffff; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .main-header { font-size: 2.8rem; font-weight: 800; color: #E50914; text-align: center; margin-bottom: 0px; letter-spacing: 1px; }
    .sub-header { font-size: 1.1rem; color: #b3b3b3; text-align: center; margin-bottom: 30px; }
    .movie-card { background-color: #1f1f1f; border: 1px solid #333333; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); transition: transform 0.2s ease; }
    .movie-card:hover { transform: translateY(-3px); border-color: #E50914; }
    .movie-title { font-size: 1.2rem; font-weight: bold; color: #ffffff; margin-bottom: 10px; }
    .badge-score { background-color: #2b2b2b; color: #46d369; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; }
    .badge-mood { background-color: #2b2b2b; color: #e50914; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; }
    .stButton>button { background-color: #E50914; color: white; border-radius: 4px; font-weight: bold; border: none; padding: 0.5rem 1rem; width: 100%; }
    .stButton>button:hover { background-color: #b20710; color: white; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-header">Bagas Pujangkoro - 22/496484/TK/54400</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Mood-Aware & Serendipity-Enhanced Film Recommendation Platform With CLAHE (Contrast Limited Adaptive Histogram Equalization)</p>',
    unsafe_allow_html=True,
)
st.divider()


# Fungsi Penerjemah dengan Caching
@st.cache_data(show_spinner=False)
def translate_to_indonesian(text):
    if not text or str(text).strip() == "" or str(text).lower() == "nan":
        return "Deskripsi film tidak tersedia."
    try:
        translator = GoogleTranslator(source="en", target="id")
        translated = translator.translate(str(text))
        return translated if translated else str(text)
    except Exception:
        return str(text)


# ==========================================
# 2. PEMUATAN DATASET & FITUR MOVIELENS
# ==========================================
@st.cache_data
def load_movielens_datasets():
    try:
        movies = pd.read_csv("movies.dat", sep="::", engine="python", encoding="latin1", names=["movieId", "title", "genres"])
        ratings = pd.read_csv("ratings.dat", sep="::", engine="python", encoding="latin1", names=["userId", "movieId", "rating", "timestamp"])
        return movies, ratings
    except Exception as e:
        st.error(f"Gagal memuat dataset (`movies.dat`, `ratings.dat`). Pastikan file berada di direktori yang sama. Detail: {e}")
        return None, None


movies_df, ratings_df = load_movielens_datasets()

if movies_df is not None and ratings_df is not None:
    movies_df.drop_duplicates(subset="movieId", inplace=True)
    ratings_df.drop_duplicates(inplace=True)
    movies_df["genres"] = movies_df["genres"].fillna("")

    # Ekstrak tahun dari judul film (misal: "Toy Story (1995)" -> 1995.0)
    movies_df["year"] = movies_df["title"].str.extract(r"\((\d{4})\)").astype(float)

    # Hitung vote_count dan vote_average langsung dari ratings.dat untuk Unpopularity Score
    vote_counts = ratings_df.groupby("movieId").size().reset_index(name="vote_count")
    avg_ratings = ratings_df.groupby("movieId")["rating"].mean().reset_index(name="vote_average")

    movies_df = pd.merge(movies_df, vote_counts, on="movieId", how="left")
    movies_df = pd.merge(movies_df, avg_ratings, on="movieId", how="left")
    movies_df["vote_count"] = movies_df["vote_count"].fillna(0)
    movies_df["vote_average"] = movies_df["vote_average"].fillna(3.0)
    movies_df["original_index"] = movies_df.index

    @st.cache_resource
    def compute_svd_and_unpopularity(ratings_df):
        user_item_matrix = ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        svd = TruncatedSVD(n_components=20, random_state=42)
        matrix_reduced = svd.fit_transform(user_item_matrix)
        matrix_reconstructed = np.dot(matrix_reduced, svd.components_)
        pred_ratings_df = pd.DataFrame(matrix_reconstructed, index=user_item_matrix.index, columns=user_item_matrix.columns)
        
        total_users = ratings_df["userId"].nunique()
        item_counts = ratings_df.groupby("movieId")["userId"].nunique()
        unpop_dict = (1 - (item_counts / total_users)).to_dict()
        return pred_ratings_df, unpop_dict

    with st.spinner("Memuat model matriks SVD & komponen unpopularity MovieLens..."):
        predicted_ratings_df, unpop_dict = compute_svd_and_unpopularity(ratings_df)

    def get_pred_rating(u_id, m_id):
        if u_id in predicted_ratings_df.index and m_id in predicted_ratings_df.columns:
            return np.clip(predicted_ratings_df.loc[u_id, m_id], 0, 5.0) / 5.0
        return 3.0 / 5.0

    # ==========================================
    # 3. SIDEBAR CONTROL PANEL
    # ==========================================
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60", use_container_width=True)
        st.markdown("### 🎛️ Control Panel Skripsi")
        st.success("Dataset Aktif: `MovieLens` (Pure)")
        
        st.markdown("#### Bobot Optimal Grid Search")
        alpha = st.number_input("Bobot Relevansi SVD (α)", value=0.9, step=0.1)
        beta = st.number_input("Bobot Ketidakpopuleran (β)", value=0.7, step=0.1)
        gamma = st.number_input("Bobot Ketidaksamaan Genre (γ)", value=0.1, step=0.1)
        mood_boost = st.number_input("Mood Booster Multiplier", value=1.15, step=0.05)

        if "detected_emotion" in st.session_state and st.session_state["detected_emotion"]:
            st.success(f"Emosi Aktif: **{st.session_state['detected_emotion']}**")

    # ==========================================
    # 4. PEMETAAN EMOSI PAUL EKMAN
    # ==========================================
    EMOTION_GENRE_MAPPING = {
        "Happy": ["Comedy", "Animation", "Family", "Music"],
        "Sad": ["Drama", "Romance", "Comedy", "Animation"],
        "Angry": ["Comedy", "Animation", "Fantasy", "Family"],
        "Disgust": ["Comedy", "Animation", "Fantasy", "Family"],
        "Fear": ["Comedy", "Animation", "Fantasy", "Family"],
        "Surprise": ["Action", "Adventure", "Sci-Fi", "Mystery"],
        "Neutral": ["Documentary", "Animation", "Family"]
    }

    # ==========================================
    # 5. FACIAL EMOTION RECOGNITION (FER) DENGAN LOW-LIGHT ENHANCEMENT
    # ==========================================
    st.markdown("### 🎥 Facial Emotion Recognition (FER)")
    st.markdown("Ambil foto wajah Anda menggunakan kamera. Sistem dilengkapi dengan penstabil pencahayaan otomatis (CLAHE) untuk kondisi minim cahaya.")

    if "detected_emotion" not in st.session_state:
        st.session_state["detected_emotion"] = "Happy"

    # Opsi Cadangan Manual untuk Antisipasi Demo Sidang
    use_manual_mode = st.checkbox("Gunakan Mode Pilih Emosi Manual (Cadangan jika ruangan minim cahaya)")

    if use_manual_mode:
        dom_emotion = st.selectbox(
            "Pilih Emosi Secara Manual:", 
            ["Happy", "Sad", "Angry", "Disgust", "Fear", "Surprise", "Neutral"],
            index=0
        )
        st.session_state["detected_emotion"] = dom_emotion
        st.info(f"Mode manual aktif. Emosi diatur ke: **{dom_emotion}**")
    else:
        col_cam1, col_cam2 = st.columns([1, 1])
        with col_cam1:
            camera_image = st.camera_input("Ambil Foto Wajah Anda")

        if camera_image is not None:
            bytes_data = camera_image.getvalue()
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            with col_cam2:
                with st.spinner("Menganalisis ekspresi wajah & optimasi pencahayaan..."):
                    try:
                        # 1. Cek tingkat kecerahan rata-rata (Grayscale)
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        avg_brightness = np.mean(gray_frame)

                        # 2. Jika minim cahaya (rata-rata < 70), terapkan CLAHE untuk mencerahkan wajah
                        if avg_brightness < 70:
                            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                            cl = clahe.apply(l)
                            limg = cv2.merge((cl, a, b))
                            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                            st.toast("⚠️ Pencahayaan rendah terdeteksi. Auto-Enhancement (CLAHE) diterapkan!", icon="💡")

                        # 3. Analisis DeepFace dengan frame yang sudah dioptimasi
                        analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                        dom_emotion = str(analysis[0]["dominant_emotion"]).capitalize()
                        
                        if dom_emotion not in EMOTION_GENRE_MAPPING:
                            dom_emotion = "Happy"
                            
                        st.session_state["detected_emotion"] = dom_emotion

                        st.markdown(
                            f"""
                            <div style="background-color: #1f1f1f; padding: 20px; border-radius: 10px; border-left: 5px solid #E50914; margin-top: 20px;">
                                <h4 style="color: #E50914; margin:0;">Analisis Berhasil!</h4>
                                <p style="margin: 5px 0 0 0; font-size: 1.1rem;">Emosi Terdeteksi: <b>{dom_emotion}</b></p>
                                <p style="margin: 0; font-size: 0.95rem; color: #aaa;">(Tingkat Keterangan Awal: {int(avg_brightness)}/255)</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Gagal mendeteksi wajah: {e}")
                        st.session_state["detected_emotion"] = "Happy"

    current_emotion = st.session_state["detected_emotion"]

    if current_emotion:
        st.divider()

        # ==========================================
        # 6. PEMILIHAN GENRE BERDASARKAN EMOSI
        # ==========================================
        st.markdown(f"### 🎞️ Genre Pilihan Berdasarkan Emosi: `{current_emotion}`")
        suggested_genres = EMOTION_GENRE_MAPPING.get(current_emotion, ["Comedy", "Animation"])
        
        selected_genre = st.selectbox("Pilih Genre Spesifik:", suggested_genres)

        st.divider()

        # ==========================================
        # 7. PILIH FILM ACUAN (SEED MOVIE)
        # ==========================================
        st.markdown("### 🍿 Pilih Film Acuan (Seed Movie)")
        
        subset_movies = movies_df[
            movies_df["genres"].astype(str).str.contains(selected_genre, case=False, na=False)
        ]
        if len(subset_movies) == 0:
            subset_movies = movies_df

        movie_list = subset_movies["title"].dropna().unique().tolist()
        selected_seed_title = st.selectbox("Pilih Film Referensi dari Dataset MovieLens:", movie_list)

        st.divider()

        # ==========================================
        # 8. CORE ALGORITHM (PUSBPG + MOOD-AWARE)
        # ==========================================
        st.markdown(f"### 🔥 Top-10 Rekomendasi Mood-Aware ({current_emotion.upper()})")

        if st.button("🚀 Jalankan Sistem Rekomendasi PUSBPG"):
            with st.spinner("Menghitung skor akhir dan menyusun peringkat rekomendasi..."):
                
                seed_row = movies_df[movies_df["title"] == selected_seed_title]
                if not seed_row.empty:
                    seed_movie_id = seed_row["movieId"].values[0]
                    seed_index = seed_row.index[0]
                else:
                    seed_movie_id = movies_df["movieId"].iloc[0]
                    seed_index = 0

                def get_genre_dissim(cand_id, seed_id):
                    if seed_id is None: return 0.5
                    c_row = movies_df[movies_df["movieId"] == cand_id]
                    s_row = movies_df[movies_df["movieId"] == seed_id]
                    if c_row.empty or s_row.empty: return 0.5
                    
                    c_genres = set([g.strip() for g in str(c_row["genres"].values[0]).replace("|", ",").split(",") if g.strip()])
                    s_genres = set([g.strip() for g in str(s_row["genres"].values[0]).replace("|", ",").split(",") if g.strip()])
                    if not c_genres or not s_genres: return 0.5
                    
                    intersection = len(c_genres.intersection(s_genres))
                    union = len(c_genres.union(s_genres))
                    j_sim = intersection / union if union > 0 else 0.0
                    return 1.0 - j_sim

                # Filter kandidat berdasarkan genre pilihan
                candidate_pool = movies_df[
                    (movies_df["original_index"] != seed_index) & 
                    (movies_df["genres"].astype(str).str.contains(selected_genre, case=False, na=False))
                ].copy()

                if len(candidate_pool) < 10:
                    candidate_pool = movies_df[movies_df["original_index"] != seed_index].copy()

                sample_user_id = 1
                allowed_genres = EMOTION_GENRE_MAPPING.get(current_emotion, [])

                scored_results = []
                for idx, row in candidate_pool.iterrows():
                    m_id = row["movieId"]
                    rel = get_pred_rating(sample_user_id, m_id)
                    unpop = unpop_dict.get(m_id, 0.5)
                    prof = get_genre_dissim(m_id, seed_movie_id)

                    # Formula Dasar PUSBPG (α, β, γ)
                    base_score = (alpha * rel) + (beta * unpop) + (gamma * prof)

                    # Mood-Aware Soft-Boosting
                    genres_str = str(row["genres"])
                    is_match = any(g.lower() in genres_str.lower() for g in allowed_genres)
                    multiplier = mood_boost if is_match else 1.0
                    final_score = base_score * multiplier

                    # Sinopsis deskriptif bawaan MovieLens/metadata
                    raw_overview = f"Film klasik/populer dari dataset MovieLens dengan genre utama: {row['genres']}."

                    scored_results.append({
                        "title": row["title"],
                        "genres": row["genres"],
                        "final_score": round(final_score, 4),
                        "mood_matched": is_match,
                        "overview": raw_overview
                    })

                # Urutkan berdasarkan skor tertinggi dan ambil Top 10
                recs_df = pd.DataFrame(scored_results).sort_values(by="final_score", ascending=False).head(10).reset_index(drop=True)
                recs_df["overview_id"] = recs_df["overview"]

                if not recs_df.empty:
                    st.success(f"Berhasil menampilkan Top-10 Rekomendasi Mood-Aware untuk emosi **{current_emotion}** berdasarkan film acuan **'{selected_seed_title}'**!")
                    st.write("")

                    for i in range(0, len(recs_df), 2):
                        col_a, col_b = st.columns(2)

                        with col_a:
                            if i < len(recs_df):
                                rec = recs_df.iloc[i]
                                match_badge = "✅ Sesuai Mood" if rec['mood_matched'] else "⚪ Standar"
                                st.markdown(
                                    f"""
                                    <div class="movie-card">
                                        <div class="movie-title">#{i+1} {rec['title']}</div>
                                        <p style="color: #aaaaaa; font-size: 0.85rem; margin-bottom: 8px;"><b>Genre:</b> {rec['genres']}</p>
                                        <p style="color: #cccccc; font-size: 0.95rem; line-height: 1.5; margin-bottom: 15px;">{rec['overview_id']}</p>
                                        <div style="display:flex; gap: 10px;">
                                            <span class="badge-score">Skor Akhir: {rec['final_score']}</span>
                                            <span class="badge-mood">{match_badge}</span>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        with col_b:
                            if i + 1 < len(recs_df):
                                rec = recs_df.iloc[i+1]
                                match_badge = "✅ Sesuai Mood" if rec['mood_matched'] else "⚪ Standar"
                                st.markdown(
                                    f"""
                                    <div class="movie-card">
                                        <div class="movie-title">#{i+2} {rec['title']}</div>
                                        <p style="color: #aaaaaa; font-size: 0.85rem; margin-bottom: 8px;"><b>Genre:</b> {rec['genres']}</p>
                                        <p style="color: #cccccc; font-size: 0.95rem; line-height: 1.5; margin-bottom: 15px;">{rec['overview_id']}</p>
                                        <div style="display:flex; gap: 10px;">
                                            <span class="badge-score">Skor Akhir: {rec['final_score']}</span>
                                            <span class="badge-mood">{match_badge}</span>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                else:
                    st.warning("Tidak ditemukan rekomendasi yang sesuai.")