# Proyek Analisis Sentimen — Ulasan Aplikasi Gojek (Google Play Store)

Submission **Analisis Sentimen** Dicoding. Pipeline: **scraping mandiri → preprocessing →
pelabelan otomatis (lexicon) → ekstraksi fitur → 3 skema pelatihan (termasuk deep learning) →
inference**.

- **Topik:** ulasan aplikasi **Gojek** (`com.gojek.app`).
- **Sumber data:** Google Play Store (API publik `google-play-scraper`, tanpa login).
- **Kelas sentimen:** `negatif`, `netral`, `positif` (3 kelas).

## Hasil

| Skema | Algoritma | Ekstraksi Fitur | Split | Train Acc | Test Acc |
|-------|-----------|-----------------|-------|-----------|----------|
| 1 | **BiLSTM** (Deep Learning, PyTorch) | Word Embedding | 80/20 | 98.90% | **85.95%** |
| 2 | **SVM** (LinearSVC) | TF-IDF | 80/20 | 99.21% | **87.28%** |
| 3 | **Logistic Regression** | TF-IDF | 70/30 | 98.15% | **85.90%** |

Ketiga skema memenuhi kriteria utama Dicoding (**akurasi testing ≥ 85%**) dengan **3 kombinasi
berbeda** (algoritma, ekstraksi fitur, pembagian data).

## Berkas Submission (4 file utama)

| Berkas | Keterangan |
|--------|-----------|
| `scraping.ipynb` | **Hanya** proses pengambilan data mentah (raw) ulasan Gojek dari Play Store → `gojek_reviews_raw.csv` |
| `notebook.ipynb` | Notebook pelatihan: pembersihan + dedup data, preprocessing, pelabelan lexicon, 3 skema, inference (sudah dijalankan, output tersimpan) |
| `requirements.txt` | Daftar dependensi |
| `gojek_reviews_raw.csv` | Dataset hasil scraping (15.000 baris mentah) |

> Lexicon (InSet & kamus slang) **diunduh otomatis dari sumbernya saat runtime** di dalam
> `notebook.ipynb`, sehingga tidak perlu disertakan sebagai file terpisah.

## Alur Singkat

1. **`scraping.ipynb`** — `google-play-scraper` (paginasi `continuation_token`), menyimpan data
   mentah apa adanya tanpa pembersihan.
2. **`notebook.ipynb`**:
   - Muat `gojek_reviews_raw.csv` → pembersihan & deduplikasi (≥ 10.000 sampel unik).
   - Preprocessing: case folding, normalisasi slang, hapus stopword (kata negasi dipertahankan).
   - Pelabelan **Lexicon InSet** (diunduh runtime) + augmentasi/adaptasi domain; penanganan negasi;
     kelas *netral* = ulasan tanpa kata opini.
   - Ekstraksi fitur: TF-IDF (model klasik) & embedding (BiLSTM).
   - 3 skema pelatihan + perbandingan.
   - **Inference** ensemble (majority voting) → output kategorikal.

## Cara Menjalankan

```bash
pip install -r requirements.txt
jupyter lab          # jalankan scraping.ipynb (opsional, data sudah tersedia) lalu notebook.ipynb
```

> `notebook.ipynb` sudah dijalankan dan seluruh output (akurasi, confusion matrix, inference)
> tersimpan, sehingga tidak perlu dijalankan ulang untuk meninjau hasil. Menjalankan notebook
> memerlukan koneksi internet untuk mengunduh lexicon.

## Catatan Etika

Data diambil melalui API publik `google-play-scraper` (tanpa kredensial/login), hanya berupa
ulasan publik aplikasi Gojek, untuk tujuan pembelajaran. Topik netral dan tidak mengandung isu
sensitif, sesuai pedoman Dicoding.
