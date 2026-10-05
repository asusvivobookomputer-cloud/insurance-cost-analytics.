# Health-Insurance Cost Analytics & Risk-Based Pricing Model

End-to-end data science project: SQL portfolio analysis → statistical testing → OLS assumption
diagnostics → leakage-free ML benchmark → business risk tiers.

**Business question.** What drives individual health-insurance claim costs, and can we predict them
accurately enough to support risk-based pricing and underwriting decisions?

**Data.** Public *Medical Cost* dataset (1,338 US policyholders: age, sex, BMI, children, smoker, region,
charges). Source: Brett Lantz, *Machine Learning with R* (downloaded automatically on first run).

## Key results

| Model (5-fold CV on train, 1,069 rows) | CV R² | CV RMSE | CV MAE |
|---|---|---|---|
| 1. OLS baseline | 0.722 | 6,328 | 4,393 |
| 2. OLS + `smoker × obese` | 0.842 | 4,759 | 2,695 |
| 3. OLS + `smoker × obese`, log target | 0.469 | 8,881 | 4,257 |
| **4. Poly(2) + RidgeCV (selected)** | **0.846** | **4,710** | **2,630** |
| 5. Random Forest | 0.834 | 4,871 | 2,785 |
| 6. Gradient Boosting | 0.827 | 4,979 | 2,866 |

Hold-out test set (268 rows, evaluated once): **R² = 0.919, RMSE = 3,409, MAE = 2,026**
(test R² is higher than CV R² by sampling variation — CV std is ±0.04; the CV figure is the more conservative estimate).

**Findings**
1. **Smoking dominates cost.** Smokers are 20% of policies but 49.5% of total cost (avg 32,050 vs 8,434).
   Mann-Whitney U: p < 0.001, rank-biserial r = −0.94 (very large effect).
2. **BMI matters only for smokers, and only above the obesity threshold (BMI ≥ 30).** Smoker + obese policyholders
   (10.8% of policies) generate 33.9% of cost (avg 41,558). This interaction lifted OLS R² from 0.72 to 0.84 —
   far more than any other change.
3. **Region and sex are not significant** (Kruskal-Wallis p = 0.23, Mann-Whitney p = 0.79) — pricing on them is unjustified by this data.
4. **Log-transforming the target did *not* fix the OLS assumptions** (residuals stay non-normal) and *hurt* accuracy on the
   currency scale (R² 0.47) because back-transforming `exp(·)` amplifies errors in the right tail. Heteroskedasticity in the baseline
   was a symptom of the omitted interaction: after adding it, Breusch-Pagan and White tests pass.
5. **Regularised polynomial model matches tree ensembles** while staying interpretable; trees overfit more (train–CV R² gap 0.07–0.10 vs 0.014).
6. **Business view:** the top 20% of policyholders by predicted risk account for ~51% of actual claims cost;
   predicted vs actual averages track closely across all deciles (`outputs/tables/risk_tier_calibration.csv`).

## Project structure

```
├── main.py                  # orchestrates the 7 stages
├── requirements.txt
└── src/
    ├── config.py            # paths, constants, WHO BMI / age-band reference data
    ├── data_loader.py       # download CSV, build SQLite DB (+2 reference tables)
    ├── sql_analysis.py      # 6 business queries (JOIN, GROUP BY/HAVING, CASE WHEN, subqueries)
    ├── preprocessing.py     # quality report, IQR/LOF outliers, winsorisation, feature engineering
    ├── eda.py               # EDA figures (training set only)
    ├── stats_tests.py       # assumption-driven test selection (Shapiro/Levene → t / MWU / ANOVA / Kruskal + post-hoc + effect sizes)
    ├── ols_diagnostics.py   # OLS + Jarque-Bera, Breusch-Pagan, White, Durbin-Watson, Breusch-Godfrey, VIF, HC3
    ├── modeling.py          # Pipelines, CV benchmark, calibration by risk decile
    └── utils.py             # timer decorator, figure/table saving
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Runtime: under a minute. Figures are written to `outputs/figures/`, tables to `outputs/tables/`.

## Methodology highlights

* **No data leakage.** The train/test split (stratified by smoker status) happens *before* any fitting. Imputers, scalers and encoders live
  inside a scikit-learn `Pipeline`, so cross-validation re-fits them on each training fold. EDA, tests and OLS diagnostics use the training set only.
  Model selection uses CV RMSE; the test set is touched once.
* **Outliers handled deliberately.** 139 high-charge IQR outliers (98% are smokers) are genuine, not errors → kept.
  A sensitivity check with winsorised BMI leaves results unchanged (R² 0.8499 vs 0.8499). `RobustScaler` is used because of BMI outliers.
* **Statistics beyond p-values.** Every test reports an effect size (Cohen's d, rank-biserial r, ε², Cramér's V) because with n > 1,000 almost everything is "significant".
* **Honest diagnostics.** Durbin-Watson / Breusch-Godfrey are shown only for completeness — rows are cross-sectional, so autocorrelation is not a real concern.
* **Dummy-variable trap avoided** (`drop_first=True` / `OneHotEncoder(drop="first")`).

## Limitations & next steps

* Residuals of the best model show a cluster of non-smokers whose costs are under-predicted by 10–20k — likely driven by unobserved
  factors (chronic conditions, claims history) absent from this dataset.
* Dataset is small and cross-sectional (single snapshot, US data); no time dimension or external validation.
* Next steps: hyper-parameter search (`GridSearchCV`) for tree models, prediction intervals, SHAP-based explanations, deployment as a small API.
