"""Predictive modelling: leakage-free pipelines, cross-validated benchmark, business calibration.

Design rules
------------
* Imputers, scalers and encoders live INSIDE a sklearn ``Pipeline``. During cross-validation they
  are re-fitted on each training fold only, so no information from validation/test rows leaks in.
* Models are ranked by cross-validated RMSE on the training set. The hold-out test set is
  evaluated once, at the very end, and never used for model selection.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler

from src.config import (BINARY, CATEGORICAL, ENGINEERED, N_FOLDS, NUMERIC, RANDOM_STATE)
from src.utils import save_fig, timer


# ----------------------------------------------------------------------------- pipelines
def build_preprocessor(numeric: list[str], binary: list[str], categorical: list[str]) -> ColumnTransformer:
    """Column-wise preprocessing.

    numeric     : median imputation -> RobustScaler (robust to the BMI outliers)
    binary      : most-frequent imputation (already 0/1)
    categorical : most-frequent imputation -> one-hot with drop='first' (no dummy-variable trap)
    """
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("scale", RobustScaler())]), numeric),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(drop="first"))]), categorical),
        ]
    )


def log_target(regressor) -> TransformedTargetRegressor:
    """Fit on log(y), but predict (and score) on the ORIGINAL currency scale."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def build_models() -> dict[str, tuple]:
    """Return ``{model_name: (estimator, feature_columns)}``.

    Every model is scored on the ORIGINAL currency scale (also the log-target ones), so the
    comparison is apples-to-apples.
    """
    raw = NUMERIC + BINARY + CATEGORICAL
    eng = NUMERIC + BINARY + ENGINEERED + CATEGORICAL
    alphas = 10 ** np.linspace(3, -3, 60)
    prep_raw = lambda: build_preprocessor(NUMERIC, BINARY, CATEGORICAL)
    prep_eng = lambda: build_preprocessor(NUMERIC, BINARY + ENGINEERED, CATEGORICAL)

    return {
        "1. OLS baseline": (
            Pipeline([("prep", prep_raw()), ("model", LinearRegression())]), raw),
        "2. OLS + smoker x obese": (
            Pipeline([("prep", prep_eng()), ("model", LinearRegression())]), eng),
        "3. OLS + smoker x obese (log target)": (
            log_target(Pipeline([("prep", prep_eng()), ("model", LinearRegression())])), eng),
        "4. Poly(2) + RidgeCV": (
            Pipeline([("prep", prep_eng()),
                      ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                      ("model", RidgeCV(alphas=alphas, cv=N_FOLDS,
                                        scoring="neg_mean_squared_error"))]), eng),
        "5. Random Forest": (
            Pipeline([("prep", prep_raw()),
                      ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                      random_state=RANDOM_STATE, n_jobs=-1))]), raw),
        "6. Gradient Boosting": (
            Pipeline([("prep", prep_raw()),
                      ("model", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                          max_depth=3, subsample=0.8,
                                                          random_state=RANDOM_STATE))]), raw),
    }


# ----------------------------------------------------------------------------- evaluation
def regression_metrics(y_true, y_pred) -> dict:
    """R2, MSE, RMSE and MAE computed together."""
    mse = mean_squared_error(y_true, y_pred)
    return {"R2": r2_score(y_true, y_pred), "MSE": mse, "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_true, y_pred)}


@timer
def benchmark(models: dict, X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Cross-validate every model on the training set, then score it once on the test set.

    Returns the comparison table and a dict of the models fitted on the full training set.
    """
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mse": "neg_mean_squared_error", "mae": "neg_mean_absolute_error"}
    rows, fitted = [], {}
    for name, (estimator, cols) in models.items():
        cvres = cross_validate(estimator, X_train[cols], y_train, cv=cv, scoring=scoring,
                               return_train_score=True)
        estimator.fit(X_train[cols], y_train)
        test = regression_metrics(y_test, estimator.predict(X_test[cols]))
        rows.append({
            "model": name,
            "train_R2": cvres["train_r2"].mean(),
            "cv_R2": cvres["test_r2"].mean(),
            "cv_R2_std": cvres["test_r2"].std(),
            "cv_RMSE": np.sqrt(-cvres["test_mse"].mean()),
            "cv_MAE": -cvres["test_mae"].mean(),
            "test_R2": test["R2"], "test_MSE": test["MSE"],
            "test_RMSE": test["RMSE"], "test_MAE": test["MAE"],
        })
        fitted[name] = estimator
    table = pd.DataFrame(rows).set_index("model")
    table["overfit_gap"] = table["train_R2"] - table["cv_R2"]
    return table, fitted


def feature_importance(fitted_model, top: int = 12) -> pd.Series:
    """Importances (tree models) or absolute coefficients (linear models) with readable names.

    Tip: for polynomial models prefer a tree model here - ``PolynomialFeatures`` duplicates 0/1
    columns (x == x^2), which makes coefficient magnitudes hard to read.
    """
    reg = fitted_model.regressor_ if hasattr(fitted_model, "regressor_") else fitted_model
    names = reg[:-1].get_feature_names_out()
    est = reg.steps[-1][1]
    values = est.feature_importances_ if hasattr(est, "feature_importances_") else np.abs(est.coef_)
    clean = [n.split("__")[-1] for n in names]
    return pd.Series(values, index=clean).sort_values(ascending=False).head(top)


# ----------------------------------------------------------------------------- business layer
def decile_calibration(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Average predicted vs actual cost per predicted-cost decile (pricing-tier view)."""
    df = pd.DataFrame({"actual": y_true.to_numpy(), "predicted": y_pred})
    df["risk_tier"] = pd.qcut(df["predicted"], 10, labels=[f"D{i}" for i in range(1, 11)])
    out = df.groupby("risk_tier", observed=True).agg(
        n=("actual", "size"), avg_predicted=("predicted", "mean"), avg_actual=("actual", "mean"))
    out["pct_of_total_actual_cost"] = out["avg_actual"] * out["n"] / (out["avg_actual"] * out["n"]).sum() * 100
    return out.round(1)


def plot_model_comparison(table: pd.DataFrame) -> None:
    """Bar chart of CV RMSE and CV R2 for every model."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    short = [m.split(". ", 1)[1] for m in table.index]
    ax[0].barh(short, table["cv_RMSE"], color="steelblue")
    ax[0].invert_yaxis()
    ax[0].set_title("Cross-validated RMSE (lower is better)")
    ax[0].set_xlabel("RMSE")
    ax[1].barh(short, table["cv_R2"], xerr=table["cv_R2_std"], color="seagreen")
    ax[1].invert_yaxis()
    ax[1].set_yticklabels([])
    ax[1].set_title("Cross-validated R2 (higher is better)")
    ax[1].set_xlabel("R2")
    plt.tight_layout()
    save_fig(fig, "10_model_comparison")


def plot_final_model(y_true: pd.Series, y_pred: np.ndarray, importances: pd.Series, title: str) -> None:
    """Predicted-vs-actual, residuals-vs-predicted and feature-importance panels."""
    resid = y_true.to_numpy() - y_pred
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4.5))
    ax[0].scatter(y_true, y_pred, alpha=0.5, s=14)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax[0].plot(lim, lim, "r--")
    ax[0].set(xlabel="Actual charges", ylabel="Predicted charges", title="Predicted vs actual (test set)")
    ax[1].scatter(y_pred, resid, alpha=0.5, s=14)
    ax[1].axhline(0, color="red", linestyle="--")
    ax[1].set(xlabel="Predicted charges", ylabel="Residual", title="Residuals (test set)")
    ax[2].barh(importances.index[::-1], importances.values[::-1], color="darkorange")
    ax[2].set_title("Random Forest feature importance")
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "11_final_model")
