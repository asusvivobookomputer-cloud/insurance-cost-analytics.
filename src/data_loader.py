"""Data acquisition and SQL layer.

The raw CSV is downloaded once, then loaded into a local SQLite database together with two
small reference tables (BMI categories and age bands). SQLite ships with Python and exposes
the same DB-API workflow as ``mysql.connector`` (connect -> execute -> fetch -> commit).
"""
import sqlite3

import pandas as pd

from src.config import AGE_BANDS, BMI_CATEGORIES, DATA_DIR, DATA_URL, DB_PATH, RAW_CSV


def download_data(force: bool = False) -> None:
    """Download the dataset to ``data/insurance.csv`` unless it already exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and not force:
        return
    print(f"Downloading dataset from {DATA_URL} ...")
    pd.read_csv(DATA_URL).to_csv(RAW_CSV, index=False)


def load_raw() -> pd.DataFrame:
    """Read the raw CSV into a DataFrame."""
    download_data()
    return pd.read_csv(RAW_CSV)


def build_database(df: pd.DataFrame) -> sqlite3.Connection:
    """Create (or rebuild) the SQLite database and return an open connection.

    Tables
    ------
    policyholders  : one row per policyholder (+ surrogate key ``policy_id``)
    bmi_categories : WHO BMI classification ranges
    age_bands      : age ranges used for pricing segments
    """
    conn = sqlite3.connect(DB_PATH)
    policyholders = df.copy()
    policyholders.insert(0, "policy_id", range(1, len(policyholders) + 1))
    policyholders.to_sql("policyholders", conn, if_exists="replace", index=False)

    pd.DataFrame(BMI_CATEGORIES, columns=["category", "min_bmi", "max_bmi"]).to_sql(
        "bmi_categories", conn, if_exists="replace", index=False
    )
    pd.DataFrame(AGE_BANDS, columns=["band", "min_age", "max_age"]).to_sql(
        "age_bands", conn, if_exists="replace", index=False
    )
    conn.commit()
    return conn
