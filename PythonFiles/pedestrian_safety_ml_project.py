
# Aaron Michelson
# Final Project — Pedestrian Safety ML Pipeline
# Updated November 2025

"""
Ingest multiple Wisconsin pedestrian-crash datasets, apply a consistent
preprocessing pipeline (standardized columns, parsed dates, derived time
fields and selected flag features), generate summary outputs, and run a
few example machine learning models (K-means, logistic regression, and a
decision tree) as a starting point for further analysis.
"""

# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ---------------------------------------
# Part 1. Load and Inspect the Datasets
# ---------------------------------------

"""
All seven datasets are loaded into a dictionary keyed by region name.
This makes it easier to apply consistent preprocessing and aggregate
results by region later in the pipeline.
"""

def load_csvs(base: Path) -> dict:
    """
    Load all regional pedestrian crash CSVs from the given base directory.

    Parameters
    ----------
    base : Path
        Directory containing all pedestrian crash CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from region name to raw DataFrame.
    """
    files = {
        "Statewide":  base / "PedestrianCrashes_Wisconsin_2001-2024.csv",
        "SW":         base / "PedestrianCrashes_Wisconsin_SWRegion_2001-2024.csv",
        "SE":         base / "PedestrianCrashes_Wisconsin_SERegion_2001-2024.csv",
        "NE":         base / "PedestrianCrashes_Wisconsin_NERegion_2001-2024.csv",
        "NC":         base / "PedestrianCrashes_Wisconsin_NCRegion_2001-2024.csv",
        "NW":         base / "PedestrianCrashes_Wisconsin_NWRegion_2001-2024.csv",
        "Milwaukee":  base / "PedestrianCrashes_Wisconsin_MilwaukeeCounty_2001-2024.csv",
    }

    dfs = {k: pd.read_csv(v, low_memory=False) for k, v in files.items()}

    # Basic sanity check: print shape for each dataset
    for k, d in dfs.items():
        print(f"{k:<10} -> {d.shape}")
    return dfs


# ---------------------------------------------
# Part 2. Clean, Filter and Transform the Data
# ---------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to a consistent format.

    Lowercases names, strips whitespace, and replaces spaces with underscores,
    which simplifies downstream column access and matching.
    """
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out


def parse_crash_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and parse a crash date column into a unified `crash_date` field.

    The function looks for columns containing 'date', 'crash', or ending in
    '_dt', and uses the first match found. If no candidate is found, `crash_date`
    is set to NaT.
    """
    out = df.copy()
    candidates = [c for c in out.columns if "date" in c or "crash" in c or c.endswith("_dt")]
    out["crash_date"] = pd.to_datetime(out[candidates[0]], errors="coerce") if candidates else pd.NaT
    return out


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive basic time-based features from `crash_date`.

    Adds:
      - year
      - month
      - day_of_week (0=Monday, 6=Sunday)
    """
    out = df.copy()
    dt = out["crash_date"]
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day_of_week"] = dt.dt.dayofweek
    return out


def add_driver_age_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create driver-age indicator variables from TEENDRVR and 65+DRVR columns.

    Adds:
      - flag_teen:   1 if TEENDRVR indicates a teen driver; else 0
      - flag_65plus: 1 if 65+DRVR indicates a 65+ driver; else 0

    Handles variants like '1', 'Y', 'YES', and defaults to 0 when the
    corresponding column is missing.
    """
    out = df.copy()

    # Map normalized keys back to original column names
    cols = {
        c.lower().strip().replace(" ", "").replace("+", "plus"): c
        for c in out.columns
    }

    teen_col = cols.get("teendrvr")
    older_col = cols.get("65plusdrvr")

    if teen_col:
        out["flag_teen"] = out[teen_col].astype(str).str.upper().isin(["1", "Y", "YES"]).astype(int)
    else:
        out["flag_teen"] = 0

    if older_col:
        out["flag_65plus"] = out[older_col].astype(str).str.upper().isin(["1", "Y", "YES"]).astype(int)
    else:
        out["flag_65plus"] = 0

    return out


def add_weekend_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an `is_weekend` indicator based on `day_of_week`.

    is_weekend = 1 for Saturday (5) or Sunday (6); otherwise 0.
    """
    out = df.copy()
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def add_severity_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary fatality flag from the injury severity field.

    Adds:
      - is_fatal = 1 if INJSVR (normalized to `injsvr`) equals 'K' (fatal)
      - is_fatal = 0 for all non-fatal or other severities

    If the severity column is not present, the flag is set to NaN so that
    rows can be filtered out in modeling steps.
    """
    out = df.copy()
    if "injsvr" in out.columns:
        out["is_fatal"] = (
            out["injsvr"].astype(str).str.strip().str.upper() == "K"
        ).astype(int)
    else:
        out["is_fatal"] = np.nan
    return out


def filter_year_range(df: pd.DataFrame, start: int = 2010, end: int = 2024) -> pd.DataFrame:
    """
    Filter records to an inclusive year range.

    Defaults to 2010–2024, reflecting the period where both fatal and
    non-fatal pedestrian crashes are consistently recorded.
    """
    out = df.copy()
    if "year" in out.columns:
        mask = (out["year"] >= start) & (out["year"] <= end)
        return out.loc[mask]
    return out


def clean_one(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a single DataFrame.

    Steps:
      1. Normalize column names
      2. Parse crash date
      3. Derive time-based fields
      4. Add driver age flags
      5. Add weekend flag
      6. Add severity flag
      7. Filter to the desired year range
    """
    out = normalize_columns(df)
    out = parse_crash_date(out)
    out = add_time_columns(out)
    out = add_driver_age_flags(out)
    out = add_weekend_flag(out)
    out = add_severity_flag(out)
    out = filter_year_range(out)
    return out


# -----------------------------------------------------------------
# Part 3. Aggregation
# -----------------------------------------------------------------

def yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year-by-year crash counts.

    Returns a DataFrame with:
      - year
      - crash_count
    sorted by year.
    """
    t = df.dropna(subset=["year"])
    grp = t.groupby("year", as_index=False).size().rename(columns={"size": "crash_count"})
    return grp.sort_values("year")


# -----------------------------------------------------------------
# Part 4. Feature Construction
# -----------------------------------------------------------------

def minimal_flag_rollups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yearly totals and teen/65+ driver participation rates.

    Returns a DataFrame with:
      - year
      - total       (# crashes)
      - teen        (# crashes involving a teen driver)
      - teen_rate   (proportion of crashes with teen drivers)
      - older65     (# crashes involving a 65+ driver)
      - older65_rate (proportion of crashes with 65+ drivers)
    """
    t = df.dropna(subset=["year"]).copy()
    t["any"] = 1
    out = (
        t.groupby("year", as_index=False)
         .agg(
             total=("any", "sum"),
             teen=("flag_teen", "sum"),
             older65=("flag_65plus", "sum")
         )
    )
    out["teen_rate"] = (out["teen"] / out["total"]).round(4)
    out["older65_rate"] = (out["older65"] / out["total"]).round(4)
    return out


# -----------------------------------------------------------------
# Part 5. Summary Statistics
# -----------------------------------------------------------------

def quick_describe(name: str, df: pd.DataFrame) -> None:
    """
    Print descriptive statistics for derived fields.

    Focuses on key engineered columns: year, month, day_of_week,
    weekend indicator, age flags, and fatality flag.
    """
    cols = [c for c in ["year", "month", "day_of_week", "is_weekend",
                        "flag_teen", "flag_65plus", "is_fatal"]
            if c in df.columns]

    print(f"\n{name} — derived fields describe():")
    if cols:
        print(df[cols].describe(include="all"))
    else:
        print("(no derived columns available)")


# -----------------------------------------------------------------
# Part 6. Exploratory Numerical Summaries
# -----------------------------------------------------------------

def print_yearly_summaries(yr_dict: dict) -> None:
    """
    Print basic statistics for yearly crash counts for each dataset.

    For each region, prints the year range, mean, minimum, and maximum
    annual crash counts.
    """
    print("\n--- Yearly Crash Count Summaries ---")
    for name, tbl in yr_dict.items():
        if tbl.empty:
            print(f"{name:<10} — no data")
            continue
        stats = tbl["crash_count"].describe()
        print(
            f"{name:<10}: years {int(tbl['year'].min())}-{int(tbl['year'].max())}, "
            f"mean={stats['mean']:.1f}, min={stats['min']:.0f}, max={stats['max']:.0f}"
        )


# -----------------------------------------------------------------
# Part 7. Dataset Integration for Modeling
# -----------------------------------------------------------------

def combine_clean_datasets(clean_dict: dict) -> pd.DataFrame:
    """
    Combine all cleaned datasets into a single DataFrame with a `region` label.

    This integrated table is convenient for statewide modeling, cross-region
    comparisons, and stratified analysis.
    """
    frames = []
    for name, df in clean_dict.items():
        tmp = df.copy()
        tmp["region"] = name
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------
# Part 8. Machine Learning Analyses
# -----------------------------------------------------------------

def run_kmeans_example(df: pd.DataFrame, n_clusters: int = 4) -> None:
    """
    Run a simple K-means clustering on selected numeric features.

    Uses:
      - year
      - is_weekend
      - flag_teen
      - flag_65plus
      - is_fatal

    Prints cluster sizes and mean fatality rate per cluster.
    """
    print("\n--- K-means clustering (illustrative) ---")

    feature_cols = [c for c in ["year", "is_weekend", "flag_teen", "flag_65plus", "is_fatal"]
                    if c in df.columns]

    if len(feature_cols) < 2:
        print("Not enough numeric features available for K-means.")
        return

    tmp = df.dropna(subset=feature_cols).copy()
    X = tmp[feature_cols]

    if X.empty:
        print("No rows available for K-means after dropping missing values.")
        return

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    tmp["cluster"] = km.fit_predict(X)

    print(f"Used features: {feature_cols}")
    summary = (
        tmp.groupby("cluster")
           .agg(count=("cluster", "size"),
                fatal_mean=("is_fatal", "mean"))
           .reset_index()
    )
    summary["fatal_mean"] = summary["fatal_mean"].round(4)
    print(summary)


def run_logistic_regression_example(df: pd.DataFrame) -> None:
    """
    Run a basic logistic regression model on selected engineered features.

    Target:
      - is_fatal

    Features (if available):
      - is_weekend
      - flag_teen
      - flag_65plus
      - year

    Prints confusion matrix, classification report, and ROC AUC.
    """
    print("\n--- Logistic regression (illustrative) ---")

    required = ["is_fatal"]
    feature_candidates = ["is_weekend", "flag_teen", "flag_65plus", "year"]

    available = [c for c in feature_candidates if c in df.columns]

    if not all(col in df.columns for col in required):
        print("Target `is_fatal` not available.")
        return

    if len(available) == 0:
        print("No usable features for logistic regression.")
        return

    tmp = df.dropna(subset=available + required).copy()
    if tmp.empty:
        print("No rows available after dropping missing values.")
        return

    X = tmp[available]
    y = tmp["is_fatal"].astype(int)

    # Handle the case where all records fall into a single class
    if y.nunique() < 2:
        print("Target variable has only one class; cannot fit model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Features used:", available)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.3f}")
    except ValueError:
        print("ROC AUC could not be computed.")


def run_decision_tree_example(df: pd.DataFrame, max_depth: int = 5) -> None:
    """
    Run a decision tree classifier using the same feature set as
    the logistic regression example.

    Prints confusion matrix, classification report, and feature importances.
    """
    print("\n--- Decision tree (illustrative) ---")

    required = ["is_fatal"]
    feature_candidates = ["is_weekend", "flag_teen", "flag_65plus", "year"]

    available = [c for c in feature_candidates if c in df.columns]

    if not all(col in df.columns for col in required):
        print("Target `is_fatal` not available.")
        return

    if len(available) == 0:
        print("No usable features for decision tree.")
        return

    tmp = df.dropna(subset=available + required).copy()
    if tmp.empty:
        print("No rows available after dropping missing values.")
        return

    X = tmp[available]
    y = tmp["is_fatal"].astype(int)

    if y.nunique() < 2:
        print("Target variable has only one class; cannot fit model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    importances = pd.Series(tree.feature_importances_, index=available)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False).round(4))


# -------------------------
# Main Driver
# -------------------------

"""
Run the full pipeline:

    - load all datasets
    - clean and filter records
    - compute yearly summaries and save them
    - print basic descriptive statistics
    - combine datasets for modeling
    - run example K-means, logistic regression, and decision tree analyses
"""

if __name__ == "__main__":
    # Input folder containing all Wisconsin pedestrian crash datasets
    base = Path(
        r"C:/Users/ajmic/OneDrive/Documents/A_School/UWM/Fall2025/"
        r"COMPSCI715_ProgrammingMachineLearning/FinalProject/DataSets"
    )

    # Output folder for all files generated by the Final Project
    output_path = Path(
        r"C:/Users/ajmic/OneDrive/Documents/A_School/UWM/Fall2025/"
        r"COMPSCI715_ProgrammingMachineLearning/FinalProject"
    )

    outdir = output_path / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # Part 1 — load & inspect
    raw = load_csvs(base)

    # Part 2 — clean datasets
    clean = {k: clean_one(v) for k, v in raw.items()}

    # Part 3 — yearly crash counts (saved to FinalProject/outputs/)
    yr = {k: yearly_counts(v) for k, v in clean.items()}
    for name, tbl in yr.items():
        tbl.to_csv(outdir / f"yearly_{name.lower()}.csv", index=False)

    # Part 4 — teen vs 65+ driver rates (statewide example)
    statewide = clean["Statewide"]
    rates = minimal_flag_rollups(statewide)
    print("\nStatewide yearly teen vs 65+ share (first few rows):")
    print(rates.head())

    # Part 5 — summary statistics
    for name, df in clean.items():
        quick_describe(name, df)

    # Part 6 — numerical summaries
    print_yearly_summaries(yr)

    # Part 7 — combine datasets into one ML-ready table
    combined = combine_clean_datasets(clean)
    print(f"\nCombined dataset shape (all regions, 2010–2024): {combined.shape}")

    # Part 8 — ML examples (using statewide data)
    run_kmeans_example(statewide, n_clusters=4)
    run_logistic_regression_example(statewide)
    run_decision_tree_example(statewide, max_depth=5)

    print(f"\nSaved outputs to: {outdir}")
