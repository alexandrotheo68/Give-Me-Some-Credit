"""Streamlit app: prediksi risiko gagal bayar (credit scoring).

Jalankan:  streamlit run app.py
Butuh:     artifacts/model.pkl (dibuat dari cell terakhir test.ipynb)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import RAW_COLS, prepare_features, predict_proba_ensemble

MODEL_PATH = Path(__file__).parent / "artifacts" / "model.pkl"

st.set_page_config(page_title="Credit Scoring", page_icon="💳", layout="centered")


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if not MODEL_PATH.exists():
    st.error(
        f"File model tidak ditemukan: `{MODEL_PATH}`.\n\n"
        "Jalankan dulu cell 'Simpan model' di akhir test.ipynb."
    )
    st.stop()

art = load_artifacts()

st.title("💳 Prediksi Risiko Gagal Bayar")
st.caption(
    "Model: Ensemble (XGBoost + LightGBM + Random Forest), dilatih pada dataset "
    "Give Me Some Credit. Output: probabilitas nasabah menunggak ≥90 hari dalam 2 tahun."
)

with st.sidebar:
    st.header("Pengaturan")
    threshold = st.slider(
        "Threshold keputusan",
        min_value=0.05, max_value=0.95,
        value=float(round(art["threshold"], 2)), step=0.01,
        help="Default = threshold optimal F1 dari validation set. "
             "Turunkan untuk menangkap lebih banyak nasabah berisiko (recall naik, "
             "tapi lebih banyak nasabah baik ikut ditolak).",
    )
    st.write(f"Threshold dari notebook: **{art['threshold']:.3f}**")


def label(p):
    return "Berisiko" if p >= threshold else "Aman"


tab_single, tab_batch = st.tabs(["Satu nasabah", "Upload CSV (banyak nasabah)"])

# ---------- Tab 1: input manual ----------
with tab_single:
    with st.form("form_nasabah"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Umur", min_value=18, max_value=110, value=45)
            income_unknown = st.checkbox("Pendapatan tidak diketahui")
            income = st.number_input("Pendapatan bulanan (MonthlyIncome)",
                                     min_value=0.0, value=5000.0, step=100.0,
                                     disabled=income_unknown)
            dep_unknown = st.checkbox("Jumlah tanggungan tidak diketahui")
            dependents = st.number_input("Jumlah tanggungan (NumberOfDependents)",
                                         min_value=0, max_value=20, value=1,
                                         disabled=dep_unknown)
            debt_ratio = st.number_input("Debt ratio (cicilan+biaya / pendapatan)",
                                         min_value=0.0, value=0.35, step=0.01)
            revolving = st.number_input(
                "Pemakaian limit kredit (RevolvingUtilization, 0-1)",
                min_value=0.0, value=0.30, step=0.01)
        with c2:
            open_lines = st.number_input("Jumlah kredit/pinjaman aktif",
                                         min_value=0, max_value=60, value=8)
            real_estate = st.number_input("Jumlah KPR / kredit properti",
                                          min_value=0, max_value=30, value=1)
            late_30 = st.number_input("Telat 30-59 hari (berapa kali)",
                                      min_value=0, max_value=20, value=0)
            late_60 = st.number_input("Telat 60-89 hari (berapa kali)",
                                      min_value=0, max_value=20, value=0)
            late_90 = st.number_input("Telat ≥90 hari (berapa kali)",
                                      min_value=0, max_value=20, value=0)
        submitted = st.form_submit_button("Prediksi", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "RevolvingUtilizationOfUnsecuredLines": revolving,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": np.nan if income_unknown else income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60,
            "NumberOfDependents": np.nan if dep_unknown else dependents,
        }])
        X = prepare_features(row, art)
        p = float(predict_proba_ensemble(X, art)[0])

        st.metric("Probabilitas gagal bayar", f"{p:.1%}")
        st.progress(min(p, 1.0))
        if p >= threshold:
            st.error(f"**Berisiko** (≥ threshold {threshold:.2f}). Disarankan review manual / tolak.")
        else:
            st.success(f"**Aman** (< threshold {threshold:.2f}).")

# ---------- Tab 2: batch CSV ----------
with tab_batch:
    st.write("Format kolom sama dengan `cs-test.csv` (kolom target boleh ada, akan diabaikan).")
    with st.expander("Kolom wajib"):
        st.code("\n".join(RAW_COLS))

    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        if df.columns[0].startswith("Unnamed"):   # kolom index seperti di cs-test.csv
            df = df.set_index(df.columns[0])
        try:
            X = prepare_features(df, art)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        proba = predict_proba_ensemble(X, art)
        out = df.copy()
        out["prob_default"] = proba
        out["prediksi"] = [label(p) for p in proba]

        n_risk = int((proba >= threshold).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total nasabah", len(out))
        c2.metric("Berisiko", n_risk)
        c3.metric("Persentase berisiko", f"{n_risk / len(out):.1%}")

        st.dataframe(out.sort_values("prob_default", ascending=False), use_container_width=True)
        st.download_button(
            "Download hasil (CSV)",
            out.to_csv().encode("utf-8"),
            file_name="hasil_prediksi.csv",
            mime="text/csv",
        )
