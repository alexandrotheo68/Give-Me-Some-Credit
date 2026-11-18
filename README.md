# 💳 Credit Scoring: Prediksi Risiko Gagal Bayar

Aplikasi web untuk memprediksi probabilitas seorang nasabah mengalami **gagal bayar serius (telat ≥90 hari) dalam 2 tahun ke depan**. Model dilatih pada dataset [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit) dan di-deploy dengan **Streamlit**.

🔗 **Demo:** https://give-me-some-credit-lr8slpkzk65dkv9rggn6t9.streamlit.app/

---

## Fitur Aplikasi

- **Prediksi satu nasabah**: isi 10 data nasabah lewat form, langsung dapat probabilitas gagal bayar dan keputusan *Aman* / *Berisiko*.
- **Prediksi banyak nasabah**: upload file CSV, lihat ringkasan jumlah nasabah berisiko, lalu download hasilnya.
- **Threshold bisa diatur**: slider di sidebar untuk menggeser batas keputusan sesuai kebijakan risiko (default = threshold optimal F1 dari validation set).
- **Nilai kosong didukung**: pendapatan dan jumlah tanggungan boleh "tidak diketahui", karena model dilatih untuk menangani data yang hilang.

## Dataset

| Item | Keterangan |
|---|---|
| Sumber | Kaggle, *Give Me Some Credit* |
| Jumlah data latih | 150.000 nasabah |
| Target | `SeriousDlqin2yrs` (1 = gagal bayar) |
| Proporsi gagal bayar | ±6,7% (data sangat tidak seimbang) |

Input yang dipakai model:

| Kolom | Arti |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Persentase pemakaian limit kartu kredit / kredit tanpa jaminan |
| `age` | Umur nasabah |
| `NumberOfTime30-59DaysPastDueNotWorse` | Berapa kali telat bayar 30–59 hari |
| `NumberOfTime60-89DaysPastDueNotWorse` | Berapa kali telat bayar 60–89 hari |
| `NumberOfTimes90DaysLate` | Berapa kali telat bayar ≥90 hari |
| `DebtRatio` | Total cicilan & biaya bulanan dibagi pendapatan bulanan |
| `MonthlyIncome` | Pendapatan bulanan |
| `NumberOfOpenCreditLinesAndLoans` | Jumlah kredit / pinjaman yang masih aktif |
| `NumberRealEstateLoansOrLines` | Jumlah kredit properti (KPR) |
| `NumberOfDependents` | Jumlah tanggungan keluarga |

## Alur Pemodelan

Seluruh proses ada di notebook `test.ipynb`:

1. **EDA**: distribusi tiap variabel, korelasi ke target, deteksi outlier dan kode anomali (96/98) di kolom keterlambatan.
2. **Preprocessing**
   - Umur < 18 dianggap tidak valid.
   - Kode 96/98 di kolom keterlambatan diberi flag lalu diganti median.
   - Missing value diisi median **dari data train** (anti-leakage).
   - Outlier di-*capping* pada persentil 99 **dari data train**.
3. **Feature engineering** (8 fitur baru): `total_late`, `weighted_late`, `has_late`, `income_per_person`, `debt_amount`, `credit_lines_per_age`, `real_estate_ratio`, `age_group`, ditambah flag `late_code_flag`, `income_missing`, `dependents_missing`.
4. **Model**: Logistic Regression, LogReg (balanced), Random Forest, XGBoost, LightGBM, LightGBM tuned (RandomizedSearchCV), dan Ensemble.
5. **Evaluasi**: AUC-ROC, serta Precision/Recall/F1 pada threshold 0,5 dan threshold optimal F1.

## Hasil

Split train/validation 80:20 (stratified).

| Model | AUC-ROC | F1@0.5 | Best Thr | Recall@opt | Precision@opt | F1@opt |
|---|---|---|---|---|---|---|
| LightGBM | 0.8701 | 0.2940 | 0.2230 | 0.4993 | 0.4069 | 0.4484 |
| **Ensemble (XGB+LGBM+RF)** | **0.8700** | **0.4358** | 0.6117 | 0.4668 | 0.4288 | 0.4470 |
| XGBoost | 0.8695 | 0.3416 | 0.7709 | 0.5282 | 0.3949 | 0.4519 |
| LightGBM (RandomizedSearchCV) | 0.8689 | 0.2816 | 0.2080 | 0.5142 | 0.3944 | 0.4464 |

**Model yang di-deploy: Ensemble** (rata-rata probabilitas XGBoost + LightGBM + Random Forest). AUC semua model hampir sama, tetapi ensemble paling stabil: F1-nya sudah baik bahkan di threshold 0,5, sehingga tidak terlalu sensitif terhadap pemilihan threshold.

## Struktur Repository

```
├── app.py               # Aplikasi Streamlit
├── preprocessing.py     # Fungsi preprocessing & prediksi ensemble
├── requirements.txt     # Daftar library (versi dikunci)
├── artifacts/
│   └── model.pkl        # Model + median + cap + threshold
└── test.ipynb           # Notebook EDA, training, evaluasi
```

## Menjalankan di Lokal

```bash
git clone https://github.com/<username>/<nama-repo>.git
cd <nama-repo>

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

> Disarankan memakai **Python 3.10**, sama dengan environment saat model dilatih.

## Melatih Ulang Model

1. Download `cs-training.csv` dan `cs-test.csv` dari Kaggle, taruh di folder root.
2. Jalankan semua cell di `test.ipynb`.
3. Cell terakhir akan menyimpan `artifacts/model.pkl`.

Kalau feature engineering di notebook diubah, ubah juga fungsi `preprocess()` di `preprocessing.py` supaya keduanya tetap identik.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Streamlit

## Keterbatasan

- Dataset berasal dari Amerika Serikat, sehingga pola risikonya belum tentu sama dengan nasabah di Indonesia.
- Precision ensemble di threshold optimal sekitar 0,43: dari setiap 100 nasabah yang ditandai berisiko, kurang lebih 43 benar-benar gagal bayar. Recall-nya sekitar 0,47, artinya kurang lebih setengah nasabah yang gagal bayar masih lolos. Aplikasi ini cocok sebagai alat bantu *screening*, bukan pengganti keputusan analis kredit.

## Author

**Alexandro Theo** — [GitHub](https://github.com/alexandrotheo68)
