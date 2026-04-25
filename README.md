# Analisis Machine Learning Transaksi Bank

Proyek machine learning komprehensif untuk menganalisis data transaksi bank menggunakan teknik clustering dan klasifikasi.

## 📋 Ringkasan Proyek

Proyek ini mengimplementasikan pipeline machine learning dua fase:
1. **Clustering** - Segmentasi pelanggan dan deteksi anomali
2. **Klasifikasi** - Klasifikasi transaksi menggunakan Decision Tree

Dataset berisi **2.512 sampel transaksi bank** dengan atribut komprehensif termasuk detail transaksi, demografi pelanggan, dan pola perilaku.

## 📊 Fitur Dataset

- **TransactionID**: Pengidentifikasi unik untuk setiap transaksi
- **AccountID**: Pengidentifikasi unik untuk setiap akun
- **TransactionAmount**: Nilai transaksi
- **TransactionDate**: Tanggal dan waktu transaksi
- **TransactionType**: Tipe transaksi (Credit atau Debit)
- **Location**: Lokasi geografis (kota di Amerika Serikat)
- **DeviceID**: Perangkat yang digunakan untuk transaksi
- **IP Address**: Alamat IPv4 saat transaksi
- **MerchantID**: Pengidentifikasi merchant unik
- **AccountBalance**: Saldo akun setelah transaksi
- **Channel**: Saluran transaksi (Online, ATM, Branch)
- **CustomerAge**: Usia pelanggan
- **CustomerOccupation**: Profesi pelanggan
- **LoginAttempts**: Jumlah upaya login sebelum transaksi
- Dan lainnya...

## 🎯 Tujuan Proyek

- Melakukan segmentasi pelanggan menggunakan algoritma clustering
- Mendeteksi anomali dalam pola transaksi
- Membangun model klasifikasi untuk kategorisasi transaksi
- Memberikan wawasan tentang perilaku pelanggan dan keamanan transaksi

## 📁 Struktur Proyek

```
bank-transaction-ml-analysis/
├── [Clustering]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb
├── [Klasifikasi]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb
├── bank_transactions_data_edited.csv
├── data_clustering.csv
├── decision_tree_model.h5
├── model_clustering.h5
└── README.md
```

## 🛠️ Teknologi yang Digunakan

- **Python 3.x**
- **Pandas** - Manipulasi dan analisis data
- **NumPy** - Komputasi numerik
- **Scikit-learn** - Algoritma machine learning
- **TensorFlow/Keras** - Model deep learning
- **Jupyter Notebook** - Analisis interaktif

## 📝 Notebook Proyek

### 1. Notebook Clustering
Melakukan segmentasi pelanggan dan deteksi anomali menggunakan berbagai teknik clustering.

**Langkah-langkah Utama:**
- Pemuatan dan eksplorasi data
- Pra-pemrosesan data
- Rekayasa fitur
- Pelatihan model clustering
- Visualisasi dan interpretasi hasil

### 2. Notebook Klasifikasi
Membangun model klasifikasi menggunakan hasil clustering.

**Langkah-langkah Utama:**
- Memuat data yang telah di-cluster
- Pembagian data (training/testing)
- Pelatihan classifier Decision Tree
- Evaluasi model
- Metrik kinerja (accuracy, precision, recall, F1-score)

## 🚀 Memulai

### Prasyarat
- Python 3.7+
- Jupyter Notebook atau JupyterLab
- Library yang diperlukan (lihat Teknologi yang Digunakan)

### Instalasi

1. Clone repository
```bash
git clone https://github.com/devin-novansyah16/bank-transaction-ml-analysis.git
cd bank-transaction-ml-analysis
```

2. Install dependensi yang diperlukan
```bash
pip install pandas numpy scikit-learn tensorflow jupyter
```

3. Jalankan Jupyter Notebook
```bash
jupyter notebook
```

4. Buka notebook secara berurutan:
   - Mulai dengan `[Clustering]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb`
   - Kemudian lanjutkan ke `[Klasifikasi]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb`

## 📈 Hasil

Proyek ini mencakup model terlatih dan hasil:
- **model_clustering.h5** - Model clustering terlatih
- **decision_tree_model.h5** - Model klasifikasi terlatih
- **data_clustering.csv** - Dataset yang telah di-cluster untuk klasifikasi

## 📧 Kontak

**Penulis:** Devin Novansyah  
**Email:** devinnovansyah1611@gmail.com

## 📄 Lisensi

Proyek ini adalah open source dan tersedia di bawah Lisensi MIT.

---

*Terakhir Diperbarui: 11 Februari 2026*
