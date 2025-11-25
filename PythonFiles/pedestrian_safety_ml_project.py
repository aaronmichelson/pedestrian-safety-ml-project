
# Aaron Michelson
# Homework #6
# November 25, 2025

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
results by region later.
"""

def load_csvs(base: Path) -> dict:
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

"""
The cleaning pipeline is modular and applied in a fixed sequence:

    normalize_columns  → consistent variable naming
    parse_crash_date   → construct a datetime column
    add_time_columns   → derive (year, month, day_of_week)
    add_driver_age_flags
    add_weekend_flag
    add_severity_flag  → binary fatality indicator
    filter_year_range  → restrict to 2010–2024

This structure makes the transformations explicit and easier to extend.
"""

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names (strip, lower, replace spaces with underscores)."""
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out


def parse_crash_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristically identify a crash date column and parse it to datetime.

    If multiple date-like columns exist, the first match is used.
    If none are found, crash_date is set to NaT.
    """
    out = df.copy()
    candidates = [c for c in out.columns if "date" in c or "crash" in c or c.endswith("_dt")]
    out["crash_date"] = pd.to_datetime(out[candidates[0]], errors="coerce") if candidates else pd.NaT
    return out


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive year, month, and day_of_week from the crash_date column."""
    out = df.copy()
    dt = out["crash_date"]
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day_of_week"] = dt.dt.dayofweek  # Mon=0..Sun=6
    return out


def add_driver_age_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create driver-age indicator variables from existing columns:
      - flag_teen:   1 if TEENDRVR indicates a teen driver; else 0
      - flag_65plus: 1 if 65+DRVR indicates a 65+ driver; else 0

    Handles variants like '1', 'Y', 'YES' as true values and defaults to 0
    if the column is not present.
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
    """Add is_weekend = 1 for Saturday/Sunday; otherwise 0."""
    out = df.copy()
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def add_severity_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary severity indicator:

      - is_fatal = 1 if INJSVR == 'K' (fatal)
      - is_fatal = 0 otherwise

    Assumes INJSVR has become `injsvr` after column normalization.
    """
    out = df.copy()
    if "injsvr" in out.columns:
        out["is_fatal"] = (
            out["injsvr"].astype(str).str.strip().str.upper() == "K"
        ).astype(int)
    else:
        # If INJSVR is missing, fill with NaN so rows can be dropped later.
        out["is_fatal"] = np.nan
    return out


def filter_year_range(df: pd.DataFrame, start: int = 2010, end: int = 2024) -> pd.DataFrame:
    """
    Filter the dataset to a given inclusive year range.

    Pre-2010 records are excluded because only fatalities were recorded,
    which would bias severity-related analysis.
    """
    out = df.copy()
    if "year" in out.columns:
        mask = (out["year"] >= start) & (out["year"] <= end)
        return out.loc[mask]
    return out


def clean_one(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a single DataFrame and
    return the cleaned result.
    """
    out = normalize_columns(df)
    out = parse_crash_date(out)
    out = add_time_columns(out)
    out = add_driver_age_flags(out)
    out = add_weekend_flag(out)
    out = add_severity_flag(out)
    out = filter_year_range(out, start=2010, end=2024)
    return out


# -----------------------------------------------------------------
# Part 3. Aggregation
# -----------------------------------------------------------------

"""
Compute annual crash counts for each dataset. This provides a simple
time-series summary that can be reused in plots or tables.
"""

def yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    t = df.dropna(subset=["year"])
    grp = t.groupby("year", as_index=False).size().rename(columns={"size": "crash_count"})
    return grp.sort_values("year")


# -----------------------------------------------------------------
# Part 4. Feature Construction
# -----------------------------------------------------------------

"""
Compute yearly rollups for age-related flags. This produces a compact table
with teen and 65+ driver participation and rates by year for the statewide
dataset.
"""

def minimal_flag_rollups(df: pd.DataFrame) -> pd.DataFrame:
    t = df.dropna(subset=["year"]).copy()
    t["any"] = 1
    out = (
        t.groupby("year", as_index=False)
         .agg(total=("any", "sum"),
              teen=("flag_teen", "sum"),
              older65=("flag_65plus", "sum"))
    )
    out["teen_rate"] = (out["teen"] / out["total"]).round(4)
    out["older65_rate"] = (out["older65"] / out["total"]).round(4)
    return out[["year", "total", "teen", "teen_rate", "older65", "older65_rate"]]


# -----------------------------------------------------------------
# Part 5. Summary Statistics
# -----------------------------------------------------------------

"""
Generate descriptive statistics for derived numeric columns (year, month,
day_of_week, and flags). This is useful for validating ranges and checking
for unexpected values.
"""

def quick_describe(name: str, df: pd.DataFrame) -> None:
    cols = [c for c in ["year", "month", "day_of_week", "is_weekend",
                        "flag_teen", "flag_65plus", "is_fatal"]
            if c in df.columns]
    print(f"\n{name} — derived fields describe():")
    if cols:
        print(df[cols].describe(include="all"))
    else:
        print("(no derived columns available to summarize)")


# -----------------------------------------------------------------
# Part 6. Exploratory Numerical Summaries
# -----------------------------------------------------------------

"""
Summarize yearly crash counts for each dataset to get a quick view of
crash frequency and variability over time.
"""

def print_yearly_summaries(yr_dict: dict) -> None:
    """Print basic statistics for yearly crash counts for each dataset."""
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

"""
Combine all cleaned datasets into a single DataFrame with a `region` label.
This supports statewide modeling as well as region-specific analysis.
"""

def combine_clean_datasets(clean_dict: dict) -> pd.DataFrame:
    frames = []
    for name, df in clean_dict.items():
        tmp = df.copy()
        tmp["region"] = name
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)
    return combined


# -----------------------------------------------------------------
# Part 8. Simple Machine Learning Analyses (Scaffolding)
# -----------------------------------------------------------------

"""
The following functions implement basic versions of three machine learning
methods using a small set of engineered features:

    - K-means clustering
    - Logistic regression
    - Decision tree classifier

These examples can be extended with additional features, tuning, and
visualization as the analysis matures.
"""

def run_kmeans_example(df: pd.DataFrame, n_clusters: int = 4) -> None:
    """
    Run K-means clustering on available numeric features and print cluster
    sizes and mean fatality rates per cluster.
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
    print("Cluster sizes and fatality shares:")
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
    Run a simple logistic regression to predict is_fatal using a small
    feature set. Prints basic classification metrics and ROC AUC.
    """
    print("\n--- Logistic regression (illustrative) ---")
    required = ["is_fatal"]
    feature_candidates = ["is_weekend", "flag_teen", "flag_65plus", "year"]

    available_features = [c for c in feature_candidates if c in df.columns]
    if not all(col in df.columns for col in required):
        print("Required target `is_fatal` not available; skipping logistic regression.")
        return

    if len(available_features) == 0:
        print("No usable features available for logistic regression.")
        return

    tmp = df.dropna(subset=available_features + required).copy()
    if tmp.empty:
        print("No rows available for logistic regression after dropping missing values.")
        return

    X = tmp[available_features]
    y = tmp["is_fatal"].astype(int)

    # Guard against degenerate target
    if y.nunique() < 2:
        print("Target variable has only one class; cannot fit logistic regression.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Features used:", available_features)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.3f}")
    except ValueError:
        print("ROC AUC could not be computed (possibly due to degenerate classes).")


def run_decision_tree_example(df: pd.DataFrame, max_depth: int = 5) -> None:
    """
    Run a simple decision tree classifier to predict is_fatal using the
    same feature set as the logistic regression example.
    """
    print("\n--- Decision tree (illustrative) ---")
    required = ["is_fatal"]
    feature_candidates = ["is_weekend", "flag_teen", "flag_65plus", "year"]

    available_features = [c for c in feature_candidates if c in df.columns]
    if not all(col in df.columns for col in required):
        print("Required target `is_fatal` not available; skipping decision tree.")
        return

    if len(available_features) == 0:
        print("No usable features available for decision tree.")
        return

    tmp = df.dropna(subset=available_features + required).copy()
    if tmp.empty:
        print("No rows available for decision tree after dropping missing values.")
        return

    X = tmp[available_features]
    y = tmp["is_fatal"].astype(int)

    if y.nunique() < 2:
        print("Target variable has only one class; cannot fit decision tree.")
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

    importances = pd.Series(tree.feature_importances_, index=available_features)
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
    # TODO: convert to a project-relative path if needed
    base = Path(
        r"C:/Users/ajmic/OneDrive/Documents/A_School/UWM/Fall2025/"
        r"COMPSCI715_ProgrammingMachineLearning/Homeworks/Homework5/Datasets/WisconsinTopsLabData"
    )

    # Part 1 — load & inspect
    raw = load_csvs(base)

    # Part 2 — clean (includes year filter and severity flag)
    clean = {k: clean_one(v) for k, v in raw.items()}

    # Part 3 — yearly counts
    outdir = base / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    yr = {k: yearly_counts(v) for k, v in clean.items()}
    for name, tbl in yr.items():
        tbl.to_csv(outdir / f"yearly_{name.lower()}.csv", index=False)

    # Part 4 — illustrative teen vs 65+ example (statewide only)
    statewide = clean["Statewide"]
    rates = minimal_flag_rollups(statewide)
    print("\nStatewide yearly teen vs 65+ share (first few rows):")
    print(rates.head())

    # Part 5 — summary stats
    for name, df in clean.items():
        quick_describe(name, df)

    # Part 6 — numerical summaries
    print_yearly_summaries(yr)

    # Part 7 — combine datasets for modeling
    combined = combine_clean_datasets(clean)
    print(f"\nCombined dataset shape (all regions, 2010–2024): {combined.shape}")

    # Part 8 — simple ML analyses (statewide example)
    run_kmeans_example(statewide, n_clusters=4)
    run_logistic_regression_example(statewide)
    run_decision_tree_example(statewide, max_depth=5)

    print(f"\nSaved outputs to: {outdir}")
