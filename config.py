"""Central configuration: paths, constants and domain knowledge used across the project."""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "insurance.csv"
DB_PATH = DATA_DIR / "insurance.db"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

# Public dataset (Brett Lantz, "Machine Learning with R"): 1,338 US health-insurance policyholders.
DATA_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
    "master/insurance.csv"
)

# ----------------------------------------------------------------------------- experiment settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
ALPHA = 0.05  # significance level for every hypothesis test
TARGET = "charges"

# ----------------------------------------------------------------------------- feature groups
NUMERIC = ["age", "bmi", "children"]
BINARY = ["sex_male", "smoker_yes"]
CATEGORICAL = ["region"]
ENGINEERED = ["obese", "smoker_obese"]

# ----------------------------------------------------------------------------- domain reference tables
# WHO adult BMI classification (lower bound inclusive, upper bound exclusive).
BMI_CATEGORIES = [
    ("Underweight", 0.0, 18.5),
    ("Normal", 18.5, 25.0),
    ("Overweight", 25.0, 30.0),
    ("Obese", 30.0, 1000.0),
]
AGE_BANDS = [
    ("18-29", 18, 30),
    ("30-39", 30, 40),
    ("40-49", 40, 50),
    ("50-64", 50, 65),
]
