"""Data quality checks, outlier handling and feature engineering.

Nothing in this module *learns* parameters from data that will later be evaluated
(scalers / imputers live inside the sklearn Pipeline in ``modeling.py``), so there is no leakage.
"""
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from src.config import BMI_CATEGORIES, ENGINEERED, NUMERIC, RANDOM_STATE
from src.utils import save_fig


# ----------------------------------------------------------------------------- data quality
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return dtype / missing / unique counts per column and save a missingno matrix."""
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isnull().sum(),
            "pct_missing": (df.isnull().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    msno.matrix(df, ax=ax, sparkline=False)
    ax.set_title("Missing-value map (white = missing)")
    save_fig(fig, "01_missing_values")
    return report


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows; return the cleaned frame and the number removed."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, before - len(cleaned)


# ----------------------------------------------------------------------------- outliers
def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences: Q1 - k*IQR and Q3 + k*IQR."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_outlier_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Count univariate IQR outliers per column."""
    rows = []
    for col in cols:
        lo, hi = iqr_bounds(df[col])
        mask = (df[col] < lo) | (df[col] > hi)
        rows.append(
            {"column": col, "lower_fence": round(lo, 2), "upper_fence": round(hi, 2),
             "n_outliers": int(mask.sum()), "pct_outliers": round(mask.mean() * 100, 2)}
        )
    return pd.DataFrame(rows).set_index("column")


def lof_outliers(df: pd.DataFrame, cols: list[str], contamination: float = 0.05) -> pd.Series:
    """Multivariate outlier flag via Local Outlier Factor (True = outlier).

    LOF is distance-based, so features are scaled first (RobustScaler). The scaler is used only
    for this exploratory flagging, never for model fitting.
    """
    X = RobustScaler().fit_transform(df[cols])
    labels = LocalOutlierFactor(n_neighbors=20, contamination=contamination).fit_predict(X)
    return pd.Series(labels == -1, index=df.index, name="lof_outlier")


def winsorize_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cap values outside the IQR fences at the fence value (keeps all rows)."""
    out = df.copy()
    for col in cols:
        lo, hi = iqr_bounds(out[col])
        out.loc[out[col] > hi, col] = hi
        out.loc[out[col] < lo, col] = lo
    return out


# ----------------------------------------------------------------------------- feature engineering
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create encoded and domain-driven features.

    * ``sex_male`` / ``smoker_yes``  - binary encodings (equivalent to one-hot with drop_first)
    * ``bmi_category``               - ORDERED categorical built from WHO BMI cut-offs
    * ``obese``                      - 1 if BMI >= 30
    * ``smoker_obese``               - interaction: smoker AND obese (the costliest segment)
    * ``log_charges``                - log-transformed target for skewed-cost modelling
    """
    out = df.copy()
    out["sex_male"] = (out["sex"] == "male").astype(int)
    out["smoker_yes"] = (out["smoker"] == "yes").astype(int)

    labels = [c[0] for c in BMI_CATEGORIES]
    edges = [c[1] for c in BMI_CATEGORIES] + [BMI_CATEGORIES[-1][2]]
    out["bmi_category"] = pd.cut(out["bmi"], bins=edges, labels=labels, right=False, ordered=True)

    out["obese"] = (out["bmi"] >= 30).astype(int)
    out["smoker_obese"] = out["smoker_yes"] * out["obese"]
    out["log_charges"] = np.log(out["charges"])
    return out


def make_ols_design(df: pd.DataFrame, engineered: bool = False) -> pd.DataFrame:
    """Design matrix for statsmodels OLS.

    ``region`` is one-hot encoded with ``drop_first=True`` to avoid the dummy-variable trap.
    """
    cols = NUMERIC + ["sex_male", "smoker_yes"] + (ENGINEERED if engineered else [])
    region = pd.get_dummies(df["region"], prefix="region", drop_first=True, dtype=int)
    return pd.concat([df[cols], region], axis=1).astype(float)
