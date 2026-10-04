"""Small helpers: timing decorator, figure saving, console formatting."""
import functools
import time

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved to disk instead of popping up
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIG_DIR, TABLE_DIR


def timer(func):
    """Decorator that prints how long the wrapped function took."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"   [timer] {func.__name__} finished in {time.perf_counter() - start:.2f}s")
        return result

    return wrapper


def set_style() -> None:
    """Apply one consistent plotting style to the whole project."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"


def save_fig(fig, name: str) -> None:
    """Save a matplotlib figure into outputs/figures and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)


def save_table(df, name: str, index: bool = True) -> None:
    """Save a DataFrame into outputs/tables as CSV."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=index)


def header(title: str) -> None:
    """Print a visible section header."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
