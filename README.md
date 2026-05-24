# Proyek Analisis Sentimen — Ulasan Aplikasi Gojek (Google Play Store)

Submission **Analisis Sentimen** Dicoding. Proyek ini membangun pipeline lengkap analisis
sentimen Bahasa Indonesia: **scraping mandiri → preprocessing → pelabelan otomatis (lexicon) →
ekstraksi fitur → 3 skema pelatihan (termasuk deep learning) → inference**.

- **Topik:** ulasan aplikasi **Gojek** (`com.gojek.app`).
- **Sumber data:** Google Play Store (API publik `google-play-scraper`, tanpa login).
- **Kelas sentimen:** `negatif`, `netral`, `positif` (3 kelas).

## Hasil

| Skema | Algoritma | Ekstraksi Fitur | Split | Train Acc | Test Acc |
|-------|-----------|-----------------|-------|-----------|----------|
| 1 | **BiLSTM** (Deep Learning, PyTorch) | Word Embedding | 80/20 | 97.03% | **86.05%** |
| 2 | **SVM** (LinearSVC) | TF-IDF | 80/20 | 99.21% | **88.09%** |
| 3 | **Logistic Regression** | TF-IDF | 70/30 | 98.09% | **86.32%** |

Ketiga skema memenuhi kriteria utama Dicoding (**akurasi testing ≥ 85%**), dengan **3 kombinasi
berbeda** (variasi algoritma, ekstraksi fitur, dan pembagian data).

## Struktur Berkas

| Berkas | Keterangan |
|--------|-----------|
| `scraping.ipynb` / `scraping.py` | Kode & proses scraping ulasan Gojek dari Play Store |
| `notebook.ipynb` | Notebook pelatihan: EDA, preprocessing, pelabelan, 3 skema, inference (sudah dijalankan, output tersimpan) |
| `dataset.csv` | Dataset hasil scraping yang sudah diberi label (10.572 sampel) |
| `requirements.txt` | Daftar dependensi |
| `data/gojek_reviews_raw.csv` | Data mentah hasil scraping (10.928 ulasan unik) |
| `lexicon/` | Lexicon InSet (`positive`/`negative`) + kamus slang (cache, agar reproducible) |
| `models/` | Artefak model terlatih (BiLSTM, SVM, Logistic Regression, vectorizer) |
| `pyproject.toml`, `uv.lock` | Konfigurasi environment (opsional, untuk pengguna `uv`) |

## Metodologi

1. **Scraping** — `google-play-scraper` dengan paginasi `continuation_token`; dedup & buang
   ulasan kosong/terlalu pendek → **10.928** ulasan unik (≥ 10.000).
2. **Preprocessing** — case folding, pembersihan (URL/mention/hashtag/karakter non-huruf),
   normalisasi *slang* (kamus *colloquial-indonesian-lexicon*), penghapusan *stopword* (Sastrawi).
   Kata **negasi** dipertahankan untuk penanganan negasi.
3. **Pelabelan otomatis (Lexicon InSet)** — skor sentimen = jumlah bobot kata; polaritas dibalik
   bila ada negasi. Lexicon **diperkaya** untuk domain ulasan aplikasi (menambah/mengoreksi kata
   sentimen umum, serta menetralkan kata domain yang netral seperti *aplikasi*, *driver*, *versi*).
   Kelas: *netral* = ulasan tanpa kata opini; *positif/negatif* berdasarkan tanda skor; ulasan
   ambigu (skor tepat 0 padahal mengandung opini) dibuang.
4. **Ekstraksi fitur** — TF-IDF (uni+bigram) untuk model klasik; *tokenizer + padding + embedding*
   untuk BiLSTM.
5. **Pelatihan** — 3 skema (lihat tabel). BiLSTM memakai *dropout*, *weight decay*, dan
   *early stopping* berbasis akurasi validasi.
6. **Inference** — prediksi *ensemble* (majority voting BiLSTM + SVM + Logistic Regression)
   menghasilkan output **kategorikal** (negatif/netral/positif). Bukti inferensi tersedia pada
   output `notebook.ipynb`.

## Cara Menjalankan

Menggunakan **uv** (direkomendasikan):

```bash
uv sync                      # buat environment dari pyproject.toml / uv.lock
uv run jupyter lab           # buka & jalankan scraping.ipynb lalu notebook.ipynb
```

Atau dengan `pip` biasa:

```bash
pip install -r requirements.txt
jupyter lab
```

> Catatan: `notebook.ipynb` sudah dijalankan dan seluruh output (akurasi, confusion matrix,
> dan hasil inference) tersimpan, sehingga tidak perlu dijalankan ulang untuk meninjau hasil.

## Catatan Etika

Data diambil melalui API publik `google-play-scraper` (tanpa kredensial/login), hanya berupa
ulasan publik aplikasi Gojek, dan digunakan untuk tujuan pembelajaran. Topik bersifat netral
dan tidak mengandung isu sensitif, sesuai pedoman Dicoding.
