# ML5_23B — Decision Tree pada Dataset Wine Quality

Repositori tugas kelompok mata kuliah Machine Learning. Kombinasi algoritma yang dipakai adalah:

**Decision Tree (supervised) + Agglomerative Clustering (unsupervised) + ResNet18 (deep learning)**

| Algoritma | Dataset | Status |
|---|---|---|
| Decision Tree | Wine Quality | selesai |
| Agglomerative Clustering | Wine Quality | dalam pengerjaan |
| ResNet18 | Imagenette 160px | dalam pengerjaan |

Repositori ini saat ini memuat pengerjaan lengkap bagian **Decision Tree**: konsep, dasar matematis, simulasi manual, kode Python, serta visualisasi dan analisis.

---

## Struktur repositori

```text
.
├── data/
│   └── raw/
│       ├── winequality-red.csv
│       ├── winequality-white.csv
│       └── winequality.names
├── src/
│   └── supervised/
│       ├── simulasi_manual_dt.py    # simulasi Gini/IG pada 8 baris
│       └── decision_tree.py        # pipeline penuh + visualisasi
├── results/
│   ├── figures/                    # 9 gambar hasil eksperimen
│   ├── metrics/                    # metrik, aturan pohon, hasil GridSearch
│   ├── models/                     # model terbaik (.joblib)
│   └── data/                       # sampel 8 baris untuk simulasi
├── reports/
│   ├── bagian3_simulasi_manual.md
│   ├── bagian4_implementasi_kode.md
│   └── bagian5_visualisasi_analisis.md
├── requirements.txt
└── README.md
```

---

## Dataset

### 1. Wine Quality (disertakan)

Dataset gabungan red dan white dari UCI Machine Learning Repository.

- Sumber: <https://archive.ics.uci.edu/dataset/186/wine+quality>
- Lisensi: CC BY 4.0
- Data mentah: 6.497 baris (1.599 red + 4.898 white), 11 fitur numerik
- Setelah 1.177 duplikat dibuang: **5.320 baris**
- Target biner: `quality ≥ 6` → baik (1), lainnya → buruk (0)
- Distribusi target: 3.332 baik (62,6%) dan 1.988 buruk (37,4%)

File CSV memakai pemisah `;`.

### 2. Imagenette 160px (tidak disertakan)

Arsip berukuran ± 94 MB sehingga tidak dimasukkan ke repositori. Unduh lalu ekstrak secara manual:

```bash
# dari root repositori
mkdir -p data/raw
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -O data/raw/imagenette2-160.tgz
tar -xzf data/raw/imagenette2-160.tgz -C data/raw/
```

Struktur yang diharapkan: `data/raw/imagenette2-160/train/` dan `data/raw/imagenette2-160/val/`, masing-masing berisi 10 folder kelas.

---

## Instalasi

Disarankan memakai virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Versi yang dipakai pada eksperimen:

| Komponen | Versi |
|---|---|
| Python | 3.14.8 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 |
| joblib | 1.6.0 |

---

## Menjalankan

### Simulasi perhitungan manual

```bash
python3 src/supervised/simulasi_manual_dt.py
```

Menghitung Gini dan information gain untuk 8 baris data secara manual, lalu memverifikasinya terhadap `DecisionTreeClassifier(max_depth=2)`. Output: tabel perhitungan, `results/data/simulasi_dt_8baris.csv`, `results/metrics/simulasi_manual_dt.json`, dan `results/figures/dt_manual_tree.png`.

### Pipeline Decision Tree

```bash
python3 src/supervised/decision_tree.py
```

Alur: muat data → deduplikasi → target biner → split 80/20 stratified → baseline → GridSearchCV 5-fold → evaluasi test → 8 visualisasi. Waktu eksekusi ± 45 detik tanpa GPU.

---

## Hasil utama

Parameter terbaik hasil `GridSearchCV` (528 kombinasi × 5 fold, skor `f1_macro`):

```text
criterion        = gini
max_depth        = 9
min_samples_leaf = 1
ccp_alpha        = 0.001
class_weight     = None
```

Pohon akhir: kedalaman 9, 35 leaf. Performa pada 1.064 baris test:

| Model | Accuracy | Balanced acc. | F1 macro | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy (mayoritas) | 0,6259 | 0,5000 | 0,3850 | 0,5000 | 0,6259 |
| Decision Tree default | 0,6880 | 0,6648 | 0,6655 | 0,6648 | 0,7181 |
| **Decision Tree tuned** | **0,7331** | **0,7206** | **0,7180** | **0,7883** | **0,8512** |

Validasi silang 5-fold model terbaik: accuracy **0,7408 ± 0,0084**, F1 macro **0,7217 ± 0,0096**.

Fitur paling berpengaruh (permutation importance): `alcohol` (0,137), `volatile acidity` (0,052), `free sulfur dioxide` (0,023).

### Pohon keputusan

![Pohon Decision Tree sampai kedalaman 2](results/figures/dt_tree_depth2.png)

Gambar lengkap dan analisisnya ada di [`reports/bagian5_visualisasi_analisis.md`](reports/bagian5_visualisasi_analisis.md).

---

## Metode ringkas

1. Data red dan white digabung, lalu **duplikat dibuang sebelum split** agar tidak terjadi kebocoran data.
2. Target dibinerkan dengan ambang `quality ≥ 6`.
3. Split 80/20 memakai `stratify=y` dan `random_state=42`.
4. Tidak ada scaling fitur karena Decision Tree tidak sensitif terhadap skala.
5. Hyperparameter dioptimalkan dengan `GridSearchCV` 5-fold dan skor `f1_macro` untuk mengatasi ketidakseimbangan kelas.
6. Evaluasi memakai accuracy, balanced accuracy, precision, recall, F1, F1 macro, ROC-AUC, PR-AUC, dan confusion matrix.

---

## Artefak hasil

| Lokasi | Isi |
|---|---|
| `results/figures/` | distribusi kelas, pohon depth 2 dan 3, kurva `max_depth`, confusion matrix, ROC/PR, feature importance, learning curve |
| `results/metrics/dt_metrics.json` | seluruh metrik dan parameter terbaik |
| `results/metrics/dt_grid_top10.csv` | 10 konfigurasi terbaik GridSearch |
| `results/metrics/dt_rules_full.txt` | seluruh aturan pohon (103 baris) |
| `results/models/decision_tree_best.joblib` | model terbaik siap dipakai |
| `results/data/simulasi_dt_8baris.csv` | data simulasi manual |

---

## Tim

| Nama | NIM | Bagian |
|---|---|---|
| _(isi nama)_ | _(isi NIM)_ | Decision Tree |
| _(isi nama)_ | _(isi NIM)_ | Agglomerative Clustering |
| _(isi nama)_ | _(isi NIM)_ | ResNet18 |
| _(isi nama)_ | _(isi NIM)_ | Integrasi laporan |

---

## Lisensi dan sumber

Kode dalam repositori ini dibuat untuk keperluan akademik. Dataset mengikuti lisensi masing-masing:

- Wine Quality — UCI Machine Learning Repository, CC BY 4.0
- Imagenette — Jeremy Howard / fast.ai, diturunkan dari ImageNet
