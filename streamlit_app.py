# ============================================================
#  streamlit_app.py  —  Aplikasi Web (Streamlit)
#  Jalankan lokal : streamlit run streamlit_app.py
#  Deploy         : upload ke https://streamlit.io/cloud
# ============================================================

import streamlit as st
from model_utils import load_model, train_model, delete_model, model_exists, predict

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title="Clickbait Detector",
    page_icon="🔍",
    layout="centered",
)

# ── CSS kustom ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1A237E, #283593);
        color: white;
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 28px;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 6px 0 0; opacity: 0.8; font-size: 0.95rem; }

    /* Card hasil */
    .result-clickbait {
        background: #FFEBEE;
        border-left: 6px solid #C62828;
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    .result-clickbait h2 { color: #C62828; margin: 0 0 6px; }
    .result-clickbait p  { color: #B71C1C; margin: 0; font-size: 0.93rem; }

    .result-nonclickbait {
        background: #E8F5E9;
        border-left: 6px solid #2E7D32;
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    .result-nonclickbait h2 { color: #2E7D32; margin: 0 0 6px; }
    .result-nonclickbait p  { color: #1B5E20; margin: 0; font-size: 0.93rem; }

    /* Metrik box */
    .metric-box {
        background: #F3F4FF;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-box .val { font-size: 1.6rem; font-weight: 700; color: #1A237E; }
    .metric-box .lbl { font-size: 0.8rem; color: #546E7A; margin-top: 2px; }

    /* Tombol streamlit */
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Muat model ke session_state ──────────────────────────────
if "model" not in st.session_state:
    st.session_state.model   = None
    st.session_state.metrics = None

if st.session_state.model is None and model_exists():
    model, metrics = load_model()
    st.session_state.model   = model
    st.session_state.metrics = metrics


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Clickbait Detector</h1>
    <p>Deteksi judul berita clickbait menggunakan Gradient Boosting + TF-IDF Character N-gram</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
#  SIDEBAR — Status model & Pelatihan
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan Model")
    st.divider()

    # Status model
    if st.session_state.model:
        st.success("✅ Model siap digunakan")
    else:
        st.error("❌ Model belum dimuat")

    st.markdown("### 📂 Latih Model Baru")
    st.caption(
        "Upload file CSV dengan kolom **headline** dan **clickbait** "
        "(0 = non-clickbait, 1 = clickbait)."
    )

    uploaded_file = st.file_uploader(
        "Pilih file dataset CSV",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Simpan sementara ke disk agar train_model bisa membacanya
        tmp_path = "uploaded_dataset.csv"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🚀 Mulai Pelatihan", use_container_width=True, type="primary"):
            log_box = st.empty()
            progress = st.progress(0)
            steps = ["Memuat dataset...", "Preprocessing...",
                     "Membagi dataset...", "Melatih model...", "Evaluasi & simpan..."]
            step_count = [0]

            def update_log(msg):
                log_box.info(msg)
                pct = min(step_count[0] / len(steps), 1.0)
                progress.progress(pct)
                step_count[0] += 1

            try:
                with st.spinner("Proses pelatihan sedang berjalan, harap tunggu..."):
                    metrics = train_model(tmp_path, progress_callback=update_log)

                st.session_state.model, st.session_state.metrics = load_model()
                progress.progress(1.0)
                log_box.empty()
                st.success("✅ Model berhasil dilatih dan disimpan!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Gagal melatih model:\n{e}")

    st.divider()

    # Hapus model
    if model_exists():
        if st.button("🗑️ Hapus Model Tersimpan",
                     use_container_width=True):
            delete_model()
            st.session_state.model   = None
            st.session_state.metrics = None
            st.success("Model telah dihapus.")
            st.rerun()

    st.divider()
    st.caption(
        "**Model:** Gradient Boosting  \n"
        "**Fitur:** TF-IDF Character N-gram  \n"
        "**ngram_range:** (3, 5)  \n"
        "**n_estimators:** 1000  \n"
        "**learning_rate:** 0.1  \n"
        "**max_depth:** 3"
    )


# ============================================================
#  AREA UTAMA — Deteksi
# ============================================================

# ── Panel metrik (hanya tampil jika model ada) ───────────────
if st.session_state.metrics:
    m = st.session_state.metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="val">{m['accuracy']}%</div>
            <div class="lbl">Accuracy</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="val">{m['precision']}%</div>
            <div class="lbl">Precision</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="val">{m['recall']}%</div>
            <div class="lbl">Recall</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="val">{m['f1']}%</div>
            <div class="lbl">F1-Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ── Input judul ──────────────────────────────────────────────
st.markdown("#### ✍️ Masukkan Judul Berita")

headline = st.text_area(
    label="Judul Berita",
    placeholder="Contoh: You Won't Believe What Happened Next...",
    height=110,
    label_visibility="collapsed"
)

col_btn, col_clear = st.columns([2, 1])
with col_btn:
    detect_clicked = st.button(
        "🔍 Deteksi Sekarang",
        use_container_width=True,
        type="primary",
        disabled=(st.session_state.model is None)
    )
with col_clear:
    clear_clicked = st.button(
        "🗑 Hapus",
        use_container_width=True,
        disabled=(st.session_state.model is None)
    )

if clear_clicked:
    st.rerun()

# ── Hasil deteksi ────────────────────────────────────────────
if detect_clicked:
    if not st.session_state.model:
        st.warning("⚠️ Model belum dimuat. Silakan latih model melalui sidebar.")
    elif not headline.strip():
        st.warning("⚠️ Silakan masukkan judul berita terlebih dahulu.")
    else:
        res = predict(st.session_state.model, headline.strip())

        if res["label"] == 1:
            st.markdown(f"""
            <div class="result-clickbait">
                <h2>⚠️ CLICKBAIT</h2>
                <p>Tingkat keyakinan model: <strong>{res['confidence']}%</strong><br>
                Judul ini terindikasi sebagai <strong>clickbait</strong>.
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-nonclickbait">
                <h2>✅ NON-CLICKBAIT</h2>
                <p>Tingkat keyakinan model: <strong>{res['confidence']}%</strong><br>
                Judul ini <strong>tidak</strong> terindikasi sebagai clickbait.
                </p>
            </div>""", unsafe_allow_html=True)

        with st.expander("🔎 Detail preprocessing"):
            st.code(f"Input asli  : {headline.strip()}\n"
                    f"Setelah preprocess : {res['text_clean']}", language="text")


# ── Info jika model belum ada ────────────────────────────────
if not st.session_state.model:
    st.info(
        "💡 **Model belum tersedia.**  \n"
        "Upload dataset CSV di sidebar kiri, lalu klik **Mulai Pelatihan** "
        "untuk melatih model. Setelah selesai, model akan otomatis tersimpan "
        "dan tidak perlu dilatih ulang di sesi berikutnya."
    )
