"""OLS regression with a full set of classical-assumption diagnostics (statsmodels)."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from src.config import ALPHA
from src.utils import save_fig


class OLSDiagnostics:
    """Fit an OLS model and test the classical assumptions.

    Assumption          Diagnostic                         Typical remedy
    ------------------  ---------------------------------  ---------------------------
    Linearity           residuals-vs-fitted plot           polynomial / interaction terms
    Normal residuals    Jarque-Bera, Q-Q plot              log-transform the target
    Homoskedasticity    Breusch-Pagan, White               log-transform / robust SE (HC3)
    No autocorrelation  Durbin-Watson, Breusch-Godfrey     HAC (Newey-West) SE
    No multicollinear.  VIF (> 5 warning, > 10 severe)     drop variable / Ridge
    """

    def __init__(self, name: str, X: pd.DataFrame, y: pd.Series):
        self.name = name
        self.X = sm.add_constant(X, has_constant="add")
        self.y = y
        self.results = sm.OLS(y, self.X).fit()

    # ------------------------------------------------------------------ diagnostics
    def vif_table(self) -> pd.Series:
        """Variance Inflation Factor for every regressor (constant excluded)."""
        cols = self.X.columns[1:]
        vif = [variance_inflation_factor(self.X.to_numpy(), i + 1) for i in range(len(cols))]
        return pd.Series(vif, index=cols, name="VIF").round(2)

    def diagnostics(self) -> dict:
        """One-row summary of fit quality and all assumption tests."""
        res = self.results
        resid, exog = res.resid, res.model.exog
        jb_p = sms.jarque_bera(resid)[1]
        bp_p = sms.het_breuschpagan(resid, exog)[1]
        with warnings.catch_warnings():
            # White's auxiliary regression contains squares of 0/1 dummies, which duplicate the
            # dummies themselves -> statsmodels warns about rank deficiency. It falls back to a
            # pseudo-inverse and the p-value remains valid, so the warning is expected.
            warnings.simplefilter("ignore")
            white_p = sms.het_white(resid, exog)[1]
        bg_p = acorr_breusch_godfrey(res, nlags=2)[1]
        return {
            "model": self.name,
            "R2": round(res.rsquared, 4),
            "Adj_R2": round(res.rsquared_adj, 4),
            "F_p": res.f_pvalue,
            "JarqueBera_p": jb_p,
            "BreuschPagan_p": bp_p,
            "White_p": white_p,
            "DurbinWatson": round(durbin_watson(resid), 3),
            "BreuschGodfrey_p": bg_p,
            "max_VIF": float(self.vif_table().max()),
            "normal_resid": bool(jb_p > ALPHA),
            "homoskedastic": bool(bp_p > ALPHA and white_p > ALPHA),
        }

    def coef_table(self, robust: bool = True, log_target: bool = False) -> pd.DataFrame:
        """Coefficient table. ``robust=True`` uses HC3 heteroskedasticity-robust standard errors.

        For a log-transformed target, ``pct_effect`` gives the approximate % change in the
        outcome for a one-unit change in the regressor: (exp(beta) - 1) * 100.
        """
        res = self.results.get_robustcov_results(cov_type="HC3") if robust else self.results
        table = pd.DataFrame(
            {"coef": res.params, "std_err": res.bse, "p_value": res.pvalues}, index=self.X.columns
        )
        if log_target:
            table["pct_effect"] = (np.exp(table["coef"]) - 1) * 100
            table.loc["const", "pct_effect"] = np.nan  # intercept has no % interpretation
        return table.round(4)

    # ------------------------------------------------------------------ plots
    def plot(self, filename: str) -> None:
        """Residuals-vs-fitted (linearity / constant variance) and Q-Q plot (normality)."""
        res = self.results
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))
        ax[0].scatter(res.fittedvalues, res.resid, alpha=0.4, s=14)
        ax[0].axhline(0, color="red", linestyle="--")
        ax[0].set_xlabel("Fitted values")
        ax[0].set_ylabel("Residuals")
        ax[0].set_title("Residuals vs fitted")
        sm.qqplot(res.resid, line="s", ax=ax[1], markersize=3)
        ax[1].set_title("Normal Q-Q plot of residuals")
        fig.suptitle(self.name, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, filename)
