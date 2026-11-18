"""Preprocessing untuk model credit scoring (Give Me Some Credit).

Fungsi preprocess() di sini HARUS identik dengan yang ada di test.ipynb.
Kalau kamu mengubah feature engineering di notebook, ubah juga di file ini,
lalu latih ulang dan simpan ulang artifacts/model.pkl.
"""
import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"

LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# 10 kolom mentah yang harus diisi user (urutan sama dengan cs-training.csv)
RAW_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def preprocess(df, medians=None, caps=None):
    df = df.copy()

    df.loc[df["age"] < 18, "age"] = np.nan

    df["late_code_flag"] = (df[LATE_COLS].isin([96, 98]).any(axis=1)).astype(int)
    for c in LATE_COLS:
        df.loc[df[c].isin([96, 98]), c] = np.nan

    df["income_missing"] = df["MonthlyIncome"].isnull().astype(int)
    df["dependents_missing"] = df["NumberOfDependents"].isnull().astype(int)

    fill_cols = ["MonthlyIncome", "NumberOfDependents", "age"] + LATE_COLS
    if medians is None:
        medians = df[fill_cols].median()
    df[fill_cols] = df[fill_cols].fillna(medians)

    cap_cols = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
    if caps is None:
        caps = {c: df[c].quantile(0.99) for c in cap_cols}
    for c in cap_cols:
        df[c] = df[c].clip(upper=caps[c])

    df["total_late"] = (df["NumberOfTime30-59DaysPastDueNotWorse"]
                        + df["NumberOfTime60-89DaysPastDueNotWorse"]
                        + df["NumberOfTimes90DaysLate"])
    df["weighted_late"] = (df["NumberOfTime30-59DaysPastDueNotWorse"] * 1
                           + df["NumberOfTime60-89DaysPastDueNotWorse"] * 2
                           + df["NumberOfTimes90DaysLate"] * 3)
    df["has_late"] = (df["total_late"] > 0).astype(int)
    df["income_per_person"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["debt_amount"] = df["DebtRatio"] * df["MonthlyIncome"]
    df["credit_lines_per_age"] = df["NumberOfOpenCreditLinesAndLoans"] / df["age"]
    df["real_estate_ratio"] = df["NumberRealEstateLoansOrLines"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 200], labels=False)

    return df, medians, caps


def prepare_features(raw_df, artifacts):
    """Data mentah -> matriks fitur siap predict, memakai median & cap dari TRAIN."""
    missing = [c for c in RAW_COLS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {missing}")

    raw = raw_df[RAW_COLS].apply(pd.to_numeric, errors="coerce")
    X, _, _ = preprocess(raw, artifacts["medians"], artifacts["caps"])
    return X[artifacts["feature_cols"]]


def predict_proba_ensemble(X, artifacts):
    """Rata-rata probabilitas XGB + LGBM2 + RF, sama seperti p_ens di notebook."""
    probs = [artifacts[name].predict_proba(X)[:, 1] for name in ("xgb", "lgbm", "rf")]
    return np.mean(probs, axis=0)
