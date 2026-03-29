# ============================================================
#  model_utils.py
#  Berisi semua fungsi inti:
#  - Preprocessing teks
#  - Pelatihan model
#  - Simpan & muat model
#
#  File ini dipakai bersama oleh app.py dan streamlit_app.py
# ============================================================

import os
import re
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Path file model yang disimpan ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(BASE_DIR, "saved_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "saved_metrics.pkl")


# ============================================================
#  PREPROCESSING
#  Sama persis dengan program penelitian (klasifikasi_remake.py)
# ============================================================

def lowercase_text(sentence: str) -> str:
    """Mengubah seluruh karakter teks menjadi huruf kecil."""
    return sentence.lower()


def normalize_text(sentence: str) -> str:
    """Menormalisasi spasi berlebih menjadi satu spasi tunggal."""
    return re.sub(r'\s+', ' ', sentence).strip()


def preprocess(text: str) -> str:
    """
    Pipeline preprocessing lengkap:
      1. Lowercase
      2. Normalisasi spasi
    """
    text = lowercase_text(text)
    text = normalize_text(text)
    return text


# ============================================================
#  TRAINING
# ============================================================

def train_model(csv_path: str, progress_callback=None) -> dict:
    """
    Melatih model dari file CSV dan menyimpannya ke disk.

    Parameters
    ----------
    csv_path         : path ke file dataset CSV
    progress_callback: fungsi(str) untuk mengirim pesan progres (opsional)

    Returns
    -------
    dict berisi metrik evaluasi (accuracy, precision, recall, f1,
    train_size, test_size), atau raise Exception jika gagal.
    """

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # 1. Muat dataset
    log("Memuat dataset...")
    df = pd.read_csv(csv_path)

    if 'headline' not in df.columns or 'clickbait' not in df.columns:
        raise ValueError(
            "Kolom 'headline' atau 'clickbait' tidak ditemukan dalam file CSV."
        )

    # 2. Preprocessing
    log("Melakukan preprocessing teks...")
    df['text_clean'] = df['headline'].apply(preprocess)

    X = df['text_clean']
    y = df['clickbait']

    # 3. Bagi dataset 80:20 dengan stratifikasi
    log("Membagi dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Bangun pipeline TF-IDF Char N-gram + Gradient Boosting
    log(f"Melatih model dengan {len(X_train)} data training...\n"
        "Proses ini membutuhkan beberapa menit, harap tunggu.")

    tfidf_char = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        min_df=5
    )
    gb_clf = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    pipeline = make_pipeline(tfidf_char, gb_clf)
    pipeline.fit(X_train, y_train)

    # 5. Evaluasi pada testing set
    log("Mengevaluasi model pada testing set...")
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy":   round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision":  round(precision_score(y_test, y_pred) * 100, 2),
        "recall":     round(recall_score(y_test, y_pred) * 100, 2),
        "f1":         round(f1_score(y_test, y_pred) * 100, 2),
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }

    # 6. Simpan model dan metrik ke disk
    log("Menyimpan model ke disk...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    with open(METRICS_PATH, 'wb') as f:
        pickle.dump(metrics, f)

    log("Model berhasil disimpan.")
    return metrics


# ============================================================
#  SIMPAN & MUAT MODEL
# ============================================================

def load_model():
    """
    Memuat model dan metrik dari disk.

    Returns
    -------
    (model, metrics) jika file tersedia, atau (None, None) jika tidak ada.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(METRICS_PATH, 'rb') as f:
            metrics = pickle.load(f)
        return model, metrics
    return None, None


def delete_model():
    """Menghapus file model dan metrik dari disk."""
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)


def model_exists() -> bool:
    """Mengecek apakah file model sudah tersedia di disk."""
    return os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)


# ============================================================
#  PREDIKSI
# ============================================================

def predict(model, text: str) -> dict:
    """
    Melakukan prediksi clickbait pada satu judul berita.

    Parameters
    ----------
    model : pipeline model yang sudah dilatih
    text  : judul berita mentah (belum di-preprocess)

    Returns
    -------
    dict dengan key:
      - label      : 1 (clickbait) atau 0 (non-clickbait)
      - confidence : persentase keyakinan model (float)
      - text_clean : teks setelah preprocessing
    """
    clean = preprocess(text)
    label = model.predict([clean])[0]
    proba = model.predict_proba([clean])[0]
    confidence = round(max(proba) * 100, 1)
    return {
        "label":      int(label),
        "confidence": confidence,
        "text_clean": clean,
    }
