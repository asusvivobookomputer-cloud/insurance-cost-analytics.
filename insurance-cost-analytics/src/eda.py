"""Exploratory data analysis plots (all computed on the TRAINING set only)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import save_fig


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Raw vs log-transformed charges: shows the right skew that motivates the log transform."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))
    sns.histplot(df["charges"], kde=True, ax=ax[0], color="steelblue")
    ax[0].set_title(f"charges (skew = {df['charges'].skew():.2f})")
    sns.histplot(df["log_charges"], kde=True, ax=ax[1], color="seagreen")
    ax[1].set_title(f"log(charges) (skew = {df['log_charges'].skew():.2f})")
    plt.tight_layout()
    save_fig(fig, "02_target_distribution")


def plot_outliers(df: pd.DataFrame) -> None:
    """Boxplots of the numeric columns, for visual outlier inspection."""
    cols = ["age", "bmi", "children", "charges"]
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 3.5))
    for a, col in zip(ax, cols):
        sns.boxplot(y=df[col], ax=a, color="lightsteelblue")
        a.set_title(col)
    plt.tight_layout()
    save_fig(fig, "03_outlier_boxplots")


def plot_group_comparisons(df: pd.DataFrame) -> None:
    """Charges by smoker status, region and BMI category x smoker."""
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4.5))
    sns.boxplot(x="smoker", y="charges", data=df, ax=ax[0])
    ax[0].set_title("Charges by smoker status")
    sns.boxplot(x="region", y="charges", data=df, ax=ax[1])
    ax[1].set_title("Charges by region")
    sns.barplot(x="bmi_category", y="charges", hue="smoker", data=df, errorbar=None, ax=ax[2])
    ax[2].set_title("Mean charges: BMI category x smoker")
    plt.tight_layout()
    save_fig(fig, "04_group_comparisons")


def plot_relationships(df: pd.DataFrame) -> None:
    """Scatter plots that reveal the smoker x BMI interaction, plus the correlation heatmap."""
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(17, 4.8))
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.6, ax=ax[0])
    ax[0].set_title("Charges vs age")
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", alpha=0.6, ax=ax[1])
    ax[1].axvline(30, color="grey", linestyle="--")
    ax[1].set_title("Charges vs BMI (dashed = obesity threshold)")
    corr = df[["age", "bmi", "children", "smoker_yes", "sex_male", "charges"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax[2])
    ax[2].set_title("Correlation matrix")
    plt.tight_layout()
    save_fig(fig, "05_relationships")
