"""Business-oriented SQL analysis on the policyholder portfolio.

Every query is a plain, readable SQL statement that uses only: GROUP BY / HAVING, JOINs on
reference tables, CASE WHEN segmentation and (nested) subqueries.
"""
import sqlite3

import pandas as pd

QUERIES: dict[str, str] = {
    # 1) How much of the total cost comes from smokers vs non-smokers?
    "01_cost_by_smoker_status": """
        SELECT smoker,
               COUNT(*)                                   AS n_policies,
               ROUND(AVG(charges), 2)                     AS avg_charges,
               ROUND(MIN(charges), 2)                     AS min_charges,
               ROUND(MAX(charges), 2)                     AS max_charges,
               ROUND(100.0 * SUM(charges) /
                     (SELECT SUM(charges) FROM policyholders), 1) AS pct_of_total_cost
        FROM policyholders
        GROUP BY smoker
        ORDER BY avg_charges DESC;
    """,
    # 2) BMI category x smoker (JOIN on a range reference table; HAVING removes tiny groups)
    "02_bmi_category_x_smoker": """
        SELECT b.category          AS bmi_category,
               p.smoker,
               COUNT(*)            AS n_policies,
               ROUND(AVG(p.charges), 2) AS avg_charges
        FROM policyholders p
        JOIN bmi_categories b
          ON p.bmi >= b.min_bmi AND p.bmi < b.max_bmi
        GROUP BY b.category, b.min_bmi, p.smoker
        HAVING COUNT(*) >= 10
        ORDER BY b.min_bmi, p.smoker;
    """,
    # 3) Age bands
    "03_cost_by_age_band": """
        SELECT a.band              AS age_band,
               COUNT(*)            AS n_policies,
               ROUND(AVG(p.charges), 2) AS avg_charges
        FROM policyholders p
        JOIN age_bands a
          ON p.age >= a.min_age AND p.age < a.max_age
        GROUP BY a.band, a.min_age
        ORDER BY a.min_age;
    """,
    # 4) Risk segmentation with CASE WHEN: share of policies vs share of total cost
    "04_risk_segments": """
        SELECT CASE
                   WHEN smoker = 'yes' AND bmi >= 30 THEN '1) Smoker + Obese'
                   WHEN smoker = 'yes'               THEN '2) Smoker, BMI < 30'
                   WHEN bmi >= 30                    THEN '3) Non-smoker, Obese'
                   ELSE                                   '4) Non-smoker, BMI < 30'
               END AS risk_segment,
               COUNT(*) AS n_policies,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM policyholders), 1) AS pct_of_policies,
               ROUND(AVG(charges), 2) AS avg_charges,
               ROUND(100.0 * SUM(charges) /
                     (SELECT SUM(charges) FROM policyholders), 1) AS pct_of_total_cost
        FROM policyholders
        GROUP BY risk_segment
        ORDER BY risk_segment;
    """,
    # 5) Subquery inside a conditional aggregate: share of above-average-cost policies per region
    "05_above_average_by_region": """
        SELECT region,
               COUNT(*) AS n_policies,
               ROUND(AVG(charges), 2) AS avg_charges,
               ROUND(100.0 * SUM(CASE WHEN charges > (SELECT AVG(charges) FROM policyholders)
                                      THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_above_portfolio_avg
        FROM policyholders
        GROUP BY region
        ORDER BY pct_above_portfolio_avg DESC;
    """,
    # 6) Cost concentration: which share of total cost comes from the most expensive 10% of policies?
    "06_top_decile_cost_share": """
        SELECT COUNT(*) AS n_policies_in_top_decile,
               ROUND(100.0 * SUM(charges) /
                     (SELECT SUM(charges) FROM policyholders), 1) AS pct_of_total_cost
        FROM policyholders
        WHERE charges >= (SELECT charges
                          FROM policyholders
                          ORDER BY charges DESC
                          LIMIT 1 OFFSET (SELECT COUNT(*) / 10 FROM policyholders));
    """,
}


def run_all(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Execute every query and return ``{query_name: result_dataframe}``."""
    return {name: pd.read_sql_query(sql, conn) for name, sql in QUERIES.items()}
