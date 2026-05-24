#!/usr/bin/env python
# coding: utf-8

# # Scraping Data Ulasan Aplikasi Gojek (Google Play Store)
# 
# **Proyek Analisis Sentimen — Dicoding**
# 
# Notebook ini melakukan **scraping mandiri** terhadap ulasan aplikasi **Gojek** (`com.gojek.app`) dari Google Play Store menggunakan pustaka [`google-play-scraper`](https://pypi.org/project/google-play-scraper/).
# 
# - **Sumber data:** Google Play Store (ulasan pengguna), bahasa Indonesia (`lang='id'`, `country='id'`).
# - **Target jumlah:** ±15.000 ulasan mentah, agar setelah pembersihan & deduplikasi tetap tersisa **≥ 10.000** sampel.
# - **Output:** `data/gojek_reviews_raw.csv`.
# 
# > Catatan etika: data diambil melalui API publik `google-play-scraper`, tanpa kredensial/login, dan hanya berupa ulasan publik. Sesuai dengan pedoman Dicoding mengenai scraping yang beretika.

# In[1]:


# Instalasi pustaka (jalankan jika belum ter-install)
# !pip install google-play-scraper pandas

import time
import pandas as pd
from google_play_scraper import reviews, Sort

print('Pustaka berhasil di-import.')


# ## 1. Konfigurasi & Fungsi Scraping
# 
# Kita mengambil ulasan secara bertahap (paginasi) menggunakan `continuation_token`. Setiap panggilan mengambil hingga 200 ulasan, lalu token dipakai untuk mengambil batch berikutnya hingga target terpenuhi.

# In[2]:


APP_ID = 'com.gojek.app'   # ID aplikasi Gojek di Google Play Store
TARGET = 15000            # target jumlah ulasan mentah
BATCH = 200               # jumlah ulasan per permintaan


def scrape_reviews(app_id, target, batch_size=200, sleep=0.4):
    """Mengambil ulasan secara bertahap memakai continuation_token."""
    collected = []
    token = None
    batch_no = 0
    while len(collected) < target:
        result, token = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=token,
        )
        if not result:
            print('Tidak ada data tambahan, berhenti.')
            break
        collected.extend(result)
        batch_no += 1
        if batch_no % 5 == 0 or len(collected) >= target:
            print(f'Batch {batch_no:>3} | total terkumpul: {len(collected)}')
        if token is None:
            print('continuation_token habis, berhenti.')
            break
        time.sleep(sleep)  # jeda sopan agar tidak membebani server
    return collected


# In[3]:


raw_reviews = scrape_reviews(APP_ID, TARGET, BATCH)
print(f'\nTotal ulasan mentah terkumpul: {len(raw_reviews)}')


# ## 2. Susun ke DataFrame & Pembersihan Awal
# 
# Kita ambil kolom yang relevan, lalu buang duplikat dan ulasan yang terlalu pendek/kosong.

# In[4]:


df = pd.DataFrame(raw_reviews)[['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'at']]
print('Ukuran sebelum pembersihan:', df.shape)
df.head()


# In[5]:


# Buang baris tanpa konten, hapus duplikat berdasarkan teks & reviewId, buang ulasan terlalu pendek
df = df.dropna(subset=['content'])
df['content'] = df['content'].astype(str).str.strip()
df = df[df['content'].str.len() >= 5]
df = df.drop_duplicates(subset=['reviewId'])
df = df.drop_duplicates(subset=['content'])
df = df.reset_index(drop=True)

print('Ukuran setelah pembersihan:', df.shape)
assert len(df) >= 10000, f'Jumlah data ({len(df)}) kurang dari 10.000! Tambah TARGET dan ulangi scraping.'
print('OK: jumlah sampel memenuhi syarat (>= 10.000).')


# In[6]:


# Distribusi rating bintang (informasi awal, label sentimen dibuat di notebook pelatihan)
print('Distribusi skor rating:')
print(df['score'].value_counts().sort_index())


# ## 3. Simpan Dataset Mentah

# In[7]:


import os
os.makedirs('data', exist_ok=True)
OUT_PATH = 'data/gojek_reviews_raw.csv'
df.to_csv(OUT_PATH, index=False)
print(f'Dataset mentah disimpan ke: {OUT_PATH}')
print(f'Jumlah baris: {len(df)} | Jumlah kolom: {df.shape[1]}')

