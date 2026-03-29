====================================================
  CLICKBAIT DETECTOR — Panduan Penggunaan
====================================================

STRUKTUR FILE
-------------
  model_utils.py    → Fungsi inti: preprocessing, training, predict
  app.py            → Aplikasi desktop (Tkinter / VSCode)
  streamlit_app.py  → Aplikasi web (Streamlit / Deploy)
  requirements.txt  → Daftar library yang dibutuhkan
  README.txt        → File ini

FILE YANG OTOMATIS DIBUAT SAAT TRAINING:
  saved_model.pkl   → Model yang sudah dilatih
  saved_metrics.pkl → Hasil evaluasi model


====================================================
  INSTALASI (lakukan sekali saja)
====================================================

  pip install -r requirements.txt


====================================================
  CARA 1 — Aplikasi Desktop (Tkinter)
====================================================

  python app.py

  - Saat pertama kali dibuka, akan muncul dialog untuk
    melatih model dari dataset CSV.
  - Setelah model tersimpan, cukup jalankan app.py
    dan model langsung dimuat otomatis.


====================================================
  CARA 2 — Aplikasi Web Lokal (Streamlit)
====================================================

  streamlit run streamlit_app.py

  - Browser akan otomatis terbuka.
  - Upload dataset CSV melalui sidebar, lalu klik
    "Mulai Pelatihan".
  - Model tersimpan otomatis dan bisa langsung dipakai.


====================================================
  CARA 3 — Deploy ke Streamlit Cloud (Online)
====================================================

  1. Buat akun di https://streamlit.io/cloud
  2. Upload semua file ke GitHub (public/private repo)
  3. Di Streamlit Cloud → New App → pilih repo kamu
  4. Isi Main file path: streamlit_app.py
  5. Klik Deploy

  CATATAN PENTING untuk Deploy:
  - Model (saved_model.pkl) tidak bisa disimpan permanen
    di Streamlit Cloud karena filesystem-nya bersifat
    sementara (reset saat restart).
  - Solusi: setiap kali deploy, upload ulang dataset CSV
    melalui sidebar dan latih ulang model.
  - Alternatif penyimpanan permanen: Google Drive / AWS S3
    (perlu modifikasi tambahan pada model_utils.py).


====================================================
  FORMAT DATASET CSV
====================================================

  File CSV harus memiliki dua kolom:
  ┌──────────────────────────────┬───────────┐
  │ headline                     │ clickbait │
  ├──────────────────────────────┼───────────┤
  │ You Won't Believe This...    │ 1         │
  │ President Signs New Bill     │ 0         │
  └──────────────────────────────┴───────────┘

  clickbait: 1 = clickbait, 0 = non-clickbait


====================================================
  SPESIFIKASI MODEL
====================================================

  Preprocessing  : Lowercase + Normalisasi spasi
  Fitur          : TF-IDF Character N-gram
  ngram_range    : (3, 5)
  min_df         : 5
  Algoritma      : Gradient Boosting (scikit-learn)
  n_estimators   : 1000
  learning_rate  : 0.1
  max_depth      : 3
  random_state   : 42
  Train/Test     : 80% / 20% (stratified)

====================================================
