# 🏦 Bank Transaction Clustering & Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)
![License](https://img.shields.io/badge/license-MIT-green)

Proyek machine learning dua tahap untuk **mengelompokkan nasabah bank** berdasarkan pola transaksi menggunakan **K-Means Clustering**, lalu **memprediksi label cluster** menggunakan model klasifikasi **Decision Tree** dan **Random Forest**.

---

## 📋 Daftar Isi

- [Latar Belakang](#-latar-belakang)
- [Struktur File](#-struktur-file)
- [Dataset](#-dataset)
- [Library yang Digunakan](#-library-yang-digunakan)
- [Tahap 1 — Clustering](#-tahap-1--clustering)
- [Tahap 2 — Klasifikasi](#-tahap-2--klasifikasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Hasil](#-hasil)
- [Author](#-author)

---

## 📌 Latar Belakang

Proyek ini terdiri dari dua tahap yang saling terhubung:

1. **Clustering** — Nasabah dikelompokkan ke dalam cluster berdasarkan kesamaan karakteristik transaksi dan profil menggunakan algoritma K-Means. Hasil clustering disimpan sebagai kolom `Target`.
2. **Klasifikasi** — Label `Target` dari hasil clustering digunakan untuk melatih model klasifikasi (Decision Tree & Random Forest), sehingga cluster nasabah baru dapat diprediksi secara otomatis.

---

## 📁 Struktur File

```
📦 bank-transaction-clustering-classification
 ┣ 📄 [Clustering]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb                              # Notebook Tahap 1: Clustering
 ┣ 📄 [klasifikasi]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb                          # Script Tahap 2: Klasifikasi
 ┣ 📄 bank_transactions_data_edited.csv                  # Dataset mentah
 ┣ 📄 data_clustering.csv                                # Output clustering (fitur numerik)
 ┣ 📄 data_clustering_inverse.csv                        # Output clustering (fitur asli/inverse)
 ┣ 📄 model_clustering.h5                                # Model KMeans tersimpan
 ┣ 📄 PCA_model_clustering.h5                            # Model KMeans + PCA (opsional)
 ┣ 📄 decision_tree_model.h5                             # Model Decision Tree tersimpan
 ┣ 📄 explore_RandomForestClassifier_classification.h5   # Model Random Forest tersimpan
 ┣ 📄 tuning_classification.h5                           # Model hasil Hyperparameter Tuning
 ┗ 📄 README.md
```

---

## 📊 Dataset

Dataset awal berisi **2.537 transaksi** dengan **16 kolom** fitur:

| Kolom | Deskripsi |
|-------|-----------|
| `TransactionID` | ID unik transaksi |
| `AccountID` | ID akun nasabah |
| `TransactionAmount` | Nilai transaksi |
| `TransactionType` | Jenis transaksi (Debit/Credit) |
| `Location` | Lokasi transaksi |
| `Channel` | Saluran transaksi (ATM/Online) |
| `CustomerAge` | Usia nasabah |
| `CustomerOccupation` | Pekerjaan nasabah |
| `TransactionDuration` | Durasi transaksi (detik) |
| `LoginAttempts` | Jumlah percobaan login |
| `AccountBalance` | Saldo rekening |
| `Target` | Label cluster hasil clustering ✅ |

Setelah preprocessing, dataset yang digunakan untuk klasifikasi berjumlah **±1.945 data**.

---

## 🔧 Library yang Digunakan

```python
# Tahap 1 - Clustering
pandas, numpy, matplotlib, seaborn
scikit-learn (LabelEncoder, StandardScaler, KMeans, PCA, silhouette_score)
yellowbrick (KElbowVisualizer)
joblib

# Tahap 2 - Klasifikasi
pandas
scikit-learn (DecisionTreeClassifier, RandomForestClassifier, RandomizedSearchCV)
joblib
```

---

## 🔵 Tahap 1 — Clustering

### Alur Pengerjaan

**1. Import Library**
Mengimpor seluruh library yang dibutuhkan untuk analisis data dan pembangunan model.

**2. Exploratory Data Analysis (EDA)**
- Menampilkan 5 baris pertama dataset (`head()`)
- Memeriksa tipe data dan jumlah baris/kolom (`info()`)
- Statistik deskriptif (`describe()`)
- Visualisasi heatmap korelasi antar fitur numerik
- Histogram distribusi setiap fitur numerik
- Boxplot nilai transaksi berdasarkan pekerjaan nasabah

**3. Pembersihan & Pra-Pemrosesan Data**
- Pengecekan dan penghapusan nilai null (`isnull().sum()`, `dropna()`)
- Pengecekan dan penghapusan data duplikat (`duplicated().sum()`, `drop_duplicates()`)
- Drop kolom tidak relevan (ID, IP Address, Date)
- Feature encoding dengan `LabelEncoder` untuk fitur kategorikal
- Handling outlier menggunakan metode IQR
- Feature scaling menggunakan `StandardScaler`
- Binning fitur `CustomerAge` menjadi 3 kelompok (Muda, Dewasa, Lansia)

**4. Membangun Model Clustering**
- Menentukan jumlah cluster optimal dengan **Elbow Method** (`KElbowVisualizer`)
- Melatih model **K-Means Clustering** dengan `n_clusters=2`
- Menghitung **Silhouette Score** untuk evaluasi model
- Visualisasi hasil clustering dalam 2D menggunakan **PCA**
- Menyimpan model dengan `joblib`

**5. Interpretasi Cluster**

Kondisi Scaled (sebelum inverse):

| Fitur | Cluster 0 (mean) | Cluster 1 (mean) |
|-------|:---:|:---:|
| TransactionAmount | -0.01 | 0.01 |
| CustomerAge | 0.02 | -0.02 |
| AccountBalance | 0.01 | -0.01 |

- **Cluster 0 — Nasabah Stabil:** Usia sedikit di atas rata-rata, saldo rekening sedikit di atas rata-rata, transaksi sedikit di bawah rata-rata. Cenderung nasabah lebih dewasa dan finansial stabil.
- **Cluster 1 — Nasabah Aktif Bertransaksi:** Usia sedikit di bawah rata-rata (lebih muda), transaksi sedikit di atas rata-rata, saldo rekening sedikit di bawah rata-rata. Cenderung nasabah muda yang aktif bertransaksi.

Kondisi Inverse (nilai asli):

| Fitur | Cluster 0 | Cluster 1 |
|-------|-----------|-----------|
| CustomerAge (mean) | ~23 tahun | ~52 tahun |
| AccountBalance (mean) | ~1.538 | ~6.397 |
| TransactionAmount (mean) | ~263 | ~247 |
| CustomerOccupation (modus) | Student | Doctor |

- **Cluster 0 — Nasabah Muda (Pelajar):** Usia 18–34 tahun, didominasi pelajar/mahasiswa, saldo rekening rendah. Rekomendasi: produk tabungan pelajar atau fitur cicilan ringan.
- **Cluster 1 — Nasabah Dewasa (Profesional):** Usia 26–80 tahun, didominasi dokter/profesional, saldo rekening tinggi. Rekomendasi: produk investasi, deposito, atau layanan wealth management.

**6. Mengeksport Data**
- Menyimpan hasil clustering ke `data_clustering.csv`
- Melakukan inverse transform untuk mengembalikan data ke nilai asli
- Menyimpan data inverse ke `data_clustering_inverse.csv`

---

## 🟠 Tahap 2 — Klasifikasi

### Alur Pipeline

```
data_clustering.csv / data_clustering_inverse.csv
   │
   ▼
(Advanced) One Hot Encoding ──► df_encoded
   │
   ▼
Data Splitting (80% train / 20% test, stratify)
   │
   ├──► Decision Tree Classifier ──► Evaluasi ──► Simpan .h5
   │
   ├──► Random Forest Classifier ──► Evaluasi ──► Simpan .h5
   │
   └──► Hyperparameter Tuning (RandomizedSearchCV) ──► Evaluasi ──► Simpan .h5
```

### Dua Skenario Dataset

| Skenario | Dataset | Keterangan |
|----------|---------|------------|
| **Standar** | `data_clustering.csv` | Fitur sudah numerik/encoded |
| **Advanced** | `data_clustering_inverse.csv` | Fitur kategorikal asli, perlu One Hot Encoding |

### Evaluasi Model

| Metrik | Deskripsi |
|--------|-----------|
| **Accuracy** | Proporsi prediksi yang benar secara keseluruhan |
| **Precision** | Ketepatan prediksi positif |
| **Recall** | Kemampuan model menemukan seluruh kasus positif |
| **F1-Score** | Rata-rata harmonik Precision dan Recall |

Tiga model dibandingkan: Decision Tree (baseline) → Random Forest (Skilled) → Random Forest + RandomizedSearchCV (Advanced).

---

## ▶️ Cara Menjalankan

**1. Clone repository ini**
```bash
git clone https://github.com/username/bank-transaction-clustering.git
cd bank-transaction-clustering
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib
```

**3. Jalankan Tahap 1 — Clustering**

Buka di Jupyter Notebook atau Google Colab:
```bash
jupyter notebook bank_clustering.ipynb
```

**4. Jalankan Tahap 2 — Klasifikasi**
```bash
python klasifikasi_clustering.py
```

> **💡 Tips:** Untuk beralih antara dataset standar dan advanced pada tahap klasifikasi, ubah baris berikut di script:
> ```python
> # Standar
> df = pd.read_csv("data_clustering.csv")
>
> # Advanced (dengan One Hot Encoding)
> df = pd.read_csv("data_clustering_inverse.csv")
> ```

---

## 📈 Hasil

### Clustering
- **Jumlah Cluster Optimal:** 2 (berdasarkan Elbow Method)
- **Silhouette Score:** ~0.573
- **Jumlah data Cluster 0:** 414 nasabah
- **Jumlah data Cluster 1:** 1.087 nasabah

### Klasifikasi
- Model dievaluasi menggunakan `classification_report` dari scikit-learn
- Model terbaik disimpan sebagai `tuning_classification.h5`

---

## 👤 Author


- **Nama:** Devin Novansyah
- **GitHub:** https://github.com/devin-novansyah16
- **LinkedIn:** [linkedin.com/in/devin-novansyah](https://www.linkedin.com/in/devin-novansyah/)
