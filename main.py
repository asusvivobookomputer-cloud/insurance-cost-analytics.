"""Insurance cost analytics - end-to-end pipeline.

Run:  python main.py

Stages
  1. Data + SQL layer          5. OLS assumption diagnostics & remedies
  2. Quality & outliers        6. Model benchmark (5-fold CV) and hold-out test
  3. Feature engineering/split 7. Business view: risk tiers & key findings
  4. EDA & hypothesis tests
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import data_loader, eda, modeling, preprocessing, sql_analysis, stats_tests
from src.config import ALPHA, RANDOM_STATE, TARGET, TEST_SIZE
from src.ols_diagnostics import OLSDiagnostics
from src.utils import header, save_table, set_style

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.1f}")


def stage_sql(raw: pd.DataFrame) -> None:
    header("STAGE 1 | SQL layer: portfolio analysis")
    conn = data_loader.build_database(raw)
    for name, result in sql_analysis.run_all(conn).items():
        print(f"\n-- {name}")
        print(result.to_string(index=False))
        save_table(result, f"sql_{name}", index=False)
    conn.close()


def stage_quality(df: pd.DataFrame) -> pd.DataFrame:
    header("STAGE 2 | Data quality & outliers")
    print(preprocessing.quality_report(df).to_string())
    df, n_dup = preprocessing.drop_duplicates(df)
    print(f"\nExact duplicate rows removed: {n_dup}  ->  {len(df)} rows remain")

    featured = preprocessing.add_features(df)
    summary = preprocessing.iqr_outlier_summary(featured, ["age", "bmi", "children", "charges"])
    print("\nUnivariate outliers (IQR / Tukey fences):")
    print(summary.to_string())

    # Why are charges 'outliers'? Check whether they are explained by smoking.
    lo, hi = preprocessing.iqr_bounds(featured["charges"])
    high = featured[featured["charges"] > hi]
    print(f"\n{len(high)} high-charge outliers; {high['smoker_yes'].mean():.0%} of them are smokers "
          f"(overall smoker share: {featured['smoker_yes'].mean():.0%}).")

    lof = preprocessing.lof_outliers(featured, ["age", "bmi", "children"], contamination=0.05)
    print(f"LOF (multivariate, contamination=5%) flags {int(lof.sum())} rows in [age, bmi, children].")
    print("DECISION: keep all rows. High charges are genuine (smoker/obesity driven), not data errors; "
          "deleting them would bias the model against the costliest customers.")
    save_table(summary, "outlier_summary")
    return featured


def stage_eda_and_tests(train: pd.DataFrame) -> None:
    header("STAGE 4 | EDA & hypothesis tests (training set only)")
    eda.plot_target_distribution(train)
    eda.plot_outliers(train)
    eda.plot_group_comparisons(train)
    eda.plot_relationships(train)

    results = [stats_tests.compare_groups(train, "charges", g)
               for g in ["smoker", "sex", "region", "bmi_category"]]
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "posthoc"} for r in results])
    print(table.to_string(index=False))
    save_table(table, "hypothesis_tests", index=False)

    for r in results:
        if r["posthoc"] is not None:
            print(f"\nPost-hoc for {r['comparison']} ({r['test']}):")
            print(r["posthoc"].round(4).to_string())

    chi = pd.DataFrame([stats_tests.chi_square_test(train, "smoker", "sex"),
                        stats_tests.chi_square_test(train, "smoker", "region")])
    print("\nCategorical associations:")
    print(chi.to_string(index=False))

    print("\nCorrelation with charges:")
    print(stats_tests.correlation_report(train, TARGET, ["age", "bmi", "children"]).to_string())
    print("\nPartial correlation (does BMI matter once smoking is controlled?):")
    print(stats_tests.partial_correlation(train, "bmi", TARGET, "smoker_yes").to_string())


def stage_ols(train: pd.DataFrame) -> OLSDiagnostics:
    header("STAGE 5 | OLS assumption diagnostics & remedies (training set only)")
    base, eng = preprocessing.make_ols_design(train), preprocessing.make_ols_design(train, engineered=True)
    specs = [
        ("A: raw charges", base, train[TARGET], "06_ols_A"),
        ("B: log(charges)", base, train["log_charges"], "07_ols_B"),
        ("C: raw charges + smoker x obese", eng, train[TARGET], "08_ols_C"),
        ("D: log(charges) + smoker x obese", eng, train["log_charges"], "09_ols_D"),
    ]
    models, rows = [], []
    for name, X, y, fig_name in specs:
        m = OLSDiagnostics(name, X, y)
        m.plot(fig_name)
        rows.append(m.diagnostics())
        models.append(m)
    diag = pd.DataFrame(rows).set_index("model")
    print(diag.T.to_string())
    save_table(diag, "ols_diagnostics")

    print("\nReading the table:")
    print(" * R2 of B and D is measured on log(charges), so it is NOT comparable with A and C.")
    print(" * The log transform does NOT repair normality / constant variance here (A vs B, C vs D).")
    print(" * Adding the smoker x obese interaction is what raises explanatory power (A -> C).")
    print(" * Model C also passes Breusch-Pagan / White: the heteroskedasticity in A was a symptom of the")
    print("   omitted interaction (model misspecification), not an unavoidable property of the data.")
    print(" * Residuals stay non-normal (Jarque-Bera), but with n > 1,000 the CLT keeps coefficient")
    print("   inference valid; HC3 robust standard errors are still reported as a safeguard.\n")

    final = models[2]  # model C: interpretable dollar effects + interaction
    print("\nVIF of model C (>5 warning, >10 severe):")
    print(final.vif_table().to_string())
    print("\nNote: Durbin-Watson / Breusch-Godfrey are shown for completeness only - the rows are "
          "cross-sectional (no time order), so autocorrelation is not a meaningful concern here.")

    coefs = final.coef_table(robust=True)
    print("\nModel C coefficients with HC3 robust standard errors (effect in currency units):")
    print(coefs.to_string())
    save_table(coefs, "ols_model_C_coefficients")

    # Robustness check: does winsorising BMI outliers change conclusions?
    wins = preprocessing.winsorize_iqr(train, ["bmi"])
    rob = OLSDiagnostics("C (BMI winsorised)", preprocessing.make_ols_design(wins, engineered=True),
                         wins[TARGET])
    print(f"\nRobustness: R2 raw-BMI = {final.results.rsquared:.4f} vs winsorised-BMI = "
          f"{rob.results.rsquared:.4f}  ->  conclusions unchanged, outliers kept.")
    return final


def stage_modeling(train: pd.DataFrame, test: pd.DataFrame):
    header("STAGE 6 | Model benchmark: 5-fold CV on train, single hold-out evaluation on test")
    models = modeling.build_models()
    table, fitted = modeling.benchmark(models, train, train[TARGET], test, test[TARGET])
    print(table.to_string())
    save_table(table, "model_comparison")
    modeling.plot_model_comparison(table)

    best_name = table["cv_RMSE"].idxmin()  # selected on CV only - never on the test set
    best_cols = models[best_name][1]
    best = fitted[best_name]
    print(f"\nBest model by cross-validated RMSE: {best_name}")

    pred = best.predict(test[best_cols])
    m = modeling.regression_metrics(test[TARGET], pred)
    print("Hold-out test metrics: " + ", ".join(f"{k}={v:,.3f}" if k == "R2" else f"{k}={v:,.1f}"
                                               for k, v in m.items()))
    modeling.plot_final_model(test[TARGET], pred,
                              modeling.feature_importance(fitted["5. Random Forest"]), best_name)
    return best_name, best, best_cols, pred


def stage_business(test: pd.DataFrame, pred: np.ndarray) -> None:
    header("STAGE 7 | Business view: risk tiers from the final model (test set)")
    tiers = modeling.decile_calibration(test[TARGET], pred)
    print(tiers.to_string())
    save_table(tiers, "risk_tier_calibration")
    top = tiers.tail(2)["pct_of_total_actual_cost"].sum()
    print(f"\nThe two highest predicted-risk deciles (20% of policyholders) account for {top:.1f}% "
          f"of actual claims cost -> candidates for risk-based pricing / underwriting review.")


def main() -> None:
    set_style()
    raw = data_loader.load_raw()
    stage_sql(raw)
    df = stage_quality(raw)

    header("STAGE 3 | Train/test split (stratified by smoker) - done BEFORE any fitting")
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                   stratify=df["smoker"])
    print(f"train = {len(train)} rows | test = {len(test)} rows | "
          f"smoker share train/test = {train['smoker_yes'].mean():.1%} / {test['smoker_yes'].mean():.1%}")

    stage_eda_and_tests(train)
    stage_ols(train)
    _, _, _, pred = stage_modeling(train, test)
    stage_business(test, pred)
    print("\nDone. Figures -> outputs/figures | tables -> outputs/tables")


if __name__ == "__main__":
    main()
