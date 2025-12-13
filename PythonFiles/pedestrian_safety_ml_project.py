
# Aaron Michelson
# Final Project — Pedestrian Safety ML Pipeline
# Updated December 2025

"""
Ingest multiple Wisconsin pedestrian-crash datasets, apply a consistent
preprocessing pipeline (standardized columns, parsed dates, derived time
fields and selected flag features), generate summary outputs, and run a
few example machine learning models (K-means, logistic regression, a
decision tree, and additional classifiers) as a starting point for
further analysis.
"""

# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

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
        "Statewide": base / "PedestrianCrashes_Wisconsin_2001-2024.csv",
        "SW": base / "PedestrianCrashes_Wisconsin_SWRegion_2001-2024.csv",
        "SE": base / "PedestrianCrashes_Wisconsin_SERegion_2001-2024.csv",
        "NE": base / "PedestrianCrashes_Wisconsin_NERegion_2001-2024.csv",
        "NC": base / "PedestrianCrashes_Wisconsin_NCRegion_2001-2024.csv",
        "NW": base / "PedestrianCrashes_Wisconsin_NWRegion_2001-2024.csv",
        "Milwaukee": base
        / "PedestrianCrashes_Wisconsin_MilwaukeeCounty_2001-2024.csv",
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
    candidates = [
        c for c in out.columns if "date" in c or "crash" in c or c.endswith("_dt")
    ]
    out["crash_date"] = (
        pd.to_datetime(out[candidates[0]], errors="coerce") if candidates else pd.NaT
    )
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


def add_time_of_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive hour-of-day and simple time-of-day categories from the crash hour.

    Expects a column like `crshhour` after normalization.

    Adds:
      - hour: integer 0–23 (NaN for invalid or missing)
      - is_night: 1 if hour in [21, 23] or [0, 5] else 0
      - is_peak:  1 if hour in {7, 8, 9, 16, 17, 18} else 0
    """
    out = df.copy()

    # After normalize_columns, CRSHHOUR -> "crshhour"
    if "crshhour" not in out.columns:
        out["hour"] = np.nan
        out["is_night"] = 0
        out["is_peak"] = 0
        return out

    h = pd.to_numeric(out["crshhour"], errors="coerce")

    # Treat obviously invalid hours as missing
    h = h.where((h >= 0) & (h <= 23), np.nan)

    out["hour"] = h

    out["is_night"] = out["hour"].isin([21, 22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    out["is_peak"] = out["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

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
    cols = {c.lower().strip().replace(" ", "").replace("+", "plus"): c for c in out.columns}

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
        out["is_fatal"] = (out["injsvr"].astype(str).str.strip().str.upper() == "K").astype(int)
    else:
        out["is_fatal"] = np.nan
    return out


def add_severity_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an ordinal injury severity variable from INJSVR (normalized to `injsvr`).

    Mapping (higher = more severe):

        K (fatal)              -> 4
        A (incapacitating)     -> 3
        B (non-incapacitating) -> 2
        C (possible injury)    -> 1
        O (property damage)    -> 0

    Any other codes or missing values become NaN.
    """
    out = df.copy()
    if "injsvr" not in out.columns:
        out["severity_ord"] = np.nan
        return out

    sev = out["injsvr"].astype(str).str.strip().str.upper()
    mapping = {"O": 0, "C": 1, "B": 2, "A": 3, "K": 4}

    out["severity_ord"] = sev.map(mapping).astype("float")
    return out


def _make_binary_flag(series: pd.Series) -> pd.Series:
    """
    Convert a column with codes like 1/0, Y/N, YES/NO, T/F, etc. into a clean 0/1 flag.

    Returns a Series of ints (0 or 1), with missing or unknown treated as 0.
    """
    s = series.astype(str).str.strip().str.upper()
    true_vals = {"1", "Y", "YES", "T", "TRUE"}
    return s.isin(true_vals).astype(int)


def add_crash_context_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary indicator variables for key crash context flags.

    Uses normalized column names such as:
      - speedflag   -> flag_speed_related
      - impaired    -> flag_impaired
      - hitrun      -> flag_hit_and_run
      - wntrroad    -> flag_winter_road
      - deerflag    -> flag_deer
      - bikeflag    -> flag_bike
      - mcycflag    -> flag_motorcycle
      - lgtrkflag   -> flag_large_truck
      - conszone    -> flag_construction_zone

    If a column is missing, the corresponding flag is set to 0.
    """
    out = df.copy()

    mapping = {
        "speedflag": "flag_speed_related",
        "impaired": "flag_impaired",
        "hitrun": "flag_hit_and_run",
        "wntrroad": "flag_winter_road",
        "deerflag": "flag_deer",
        "bikeflag": "flag_bike",
        "mcycflag": "flag_motorcycle",
        "lgtrkflag": "flag_large_truck",
        "conszone": "flag_construction_zone",
    }

    for src_col, new_flag in mapping.items():
        if src_col in out.columns:
            out[new_flag] = _make_binary_flag(out[src_col])
        else:
            out[new_flag] = 0

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
      3. Derive time-based fields (year, month, day_of_week)
      4. Add time-of-day features (hour, is_night, is_peak)
      5. Add driver age flags
      6. Add weekend flag
      7. Add severity flags (binary fatal + ordinal)
      8. Add crash context flags (speed, impairment, hit-and-run, etc.)
      9. Filter to the desired year range
    """
    out = normalize_columns(df)
    out = parse_crash_date(out)
    out = add_time_columns(out)
    out = add_time_of_day_features(out)
    out = add_driver_age_flags(out)
    out = add_weekend_flag(out)
    out = add_severity_flag(out)
    out = add_severity_ordinal(out)
    out = add_crash_context_flags(out)
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
        .agg(total=("any", "sum"), teen=("flag_teen", "sum"), older65=("flag_65plus", "sum"))
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
    weekend indicator, age flags, fatality flag, and a few context flags.
    """
    cols = [
        c
        for c in [
            "year",
            "month",
            "day_of_week",
            "hour",
            "is_weekend",
            "is_night",
            "is_peak",
            "flag_teen",
            "flag_65plus",
            "is_fatal",
            "severity_ord",
        ]
        if c in df.columns
    ]

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
# Part 8. Machine Learning Analyses (Original Examples)
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

    feature_cols = [c for c in ["year", "is_weekend", "flag_teen", "flag_65plus", "is_fatal"] if c in df.columns]

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
        .agg(count=("cluster", "size"), fatal_mean=("is_fatal", "mean"))
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


# -----------------------------------------------------------------
# Part 9. ML Utilities and Expanded Analyses
# -----------------------------------------------------------------


def build_model_dataset(
    df: pd.DataFrame,
    use_context_flags: bool = True,
    restrict_to_fatal_binary: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct an ML-ready (X, y) pair from a cleaned crash DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of clean_one or combine_clean_datasets).
    use_context_flags : bool, default True
        Whether to include crash context flags (speed, impairment, etc.) as features.
    restrict_to_fatal_binary : bool, default True
        If True, target is `is_fatal`. If False, uses `severity_ord` as target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector (binary or ordinal, depending on parameters).
    """
    if restrict_to_fatal_binary:
        target_col = "is_fatal"
        if target_col not in df.columns:
            raise ValueError("Target column 'is_fatal' not found in DataFrame.")
    else:
        target_col = "severity_ord"
        if target_col not in df.columns:
            raise ValueError("Target column 'severity_ord' not found in DataFrame.")

    base_features = [
        "year",
        "month",
        "day_of_week",
        "is_weekend",
        "hour",
        "is_night",
        "is_peak",
        "flag_teen",
        "flag_65plus",
    ]

    context_features = (
        [
            "flag_speed_related",
            "flag_impaired",
            "flag_hit_and_run",
            "flag_winter_road",
            "flag_deer",
            "flag_bike",
            "flag_motorcycle",
            "flag_large_truck",
            "flag_construction_zone",
        ]
        if use_context_flags
        else []
    )

    all_candidates = base_features + context_features
    feature_cols = [c for c in all_candidates if c in df.columns]

    if not feature_cols:
        raise ValueError("No feature columns available for modeling.")

    tmp = df.dropna(subset=[target_col] + feature_cols).copy()
    if tmp.empty:
        raise ValueError("No rows available after dropping missing values.")

    X = tmp[feature_cols]
    y = tmp[target_col]
    return X, y


def evaluate_binary_classifier(
    name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    positive_label: int = 1,
) -> dict:
    """
    Fit a binary classifier, generate predictions, and compute key metrics.

    Returns a dictionary with:
      - model_name
      - accuracy
      - precision
      - recall
      - f1
      - roc_auc (if available, else NaN)
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        try:
            scores = model.decision_function(X_test)
            auc = roc_auc_score(y_test, scores)
        except Exception:
            auc = np.nan

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        pos_label=positive_label,
        average="binary",
        zero_division=0,
    )

    return {"model_name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}


def compare_classifiers_on_fatality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train and compare multiple classifiers on the binary fatality outcome.

    Models include:
      - Logistic Regression (baseline)
      - Logistic Regression (class_weight='balanced')
      - Random Forest (class_weight='balanced')
      - Gradient Boosting

    Returns
    -------
    pd.DataFrame
        One row per model with accuracy, precision, recall, F1, and ROC AUC.
    """
    print("\n--- Model comparison on `is_fatal` (binary) ---")

    try:
        X, y = build_model_dataset(df, use_context_flags=True, restrict_to_fatal_binary=True)
    except ValueError as e:
        print(f"Cannot build dataset: {e}")
        return pd.DataFrame()

    y = y.astype(int)
    if y.nunique() < 2:
        print("Target variable has only one class; cannot compare models.")
        return pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = [
        ("Logistic (baseline)", LogisticRegression(max_iter=1000)),
        ("Logistic (balanced)", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("RandomForest (balanced)",
         RandomForestClassifier(
             n_estimators=200,
             max_depth=None,
             min_samples_split=5,
             class_weight="balanced",
             random_state=42,
             n_jobs=-1,
         )),
        ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
    ]

    results = []
    for name, model in models:
        print(f"\nFitting {name} ...")
        metrics = evaluate_binary_classifier(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    res_df = pd.DataFrame(results)
    print("\nModel comparison (higher is better for all metrics):")
    print(res_df.round(4).sort_values("recall", ascending=False))
    return res_df


def run_multiclass_severity_tree(df: pd.DataFrame, max_depth: int = 6) -> None:
    """
    Train a decision tree to predict the ordinal injury severity level (0–4).

    Uses the same engineered features as the binary fatality models, but
    treats severity as a 5-class classification problem.
    """
    print("\n--- Multiclass severity decision tree (ordinal severity_ord) ---")

    try:
        X, y = build_model_dataset(df, use_context_flags=True, restrict_to_fatal_binary=False)
    except ValueError as e:
        print(f"Cannot build dataset: {e}")
        return

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask].astype(int)

    if y.nunique() < 2:
        print("Not enough distinct severity classes to train a model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    print("\nSeverity classification report (0=PDO, 4=fatal):")
    print(classification_report(y_test, y_pred, digits=3))


def run_kmeans_with_context(df: pd.DataFrame, n_clusters: int = 5) -> None:
    """
    Run a K-means clustering experiment using extended engineered features.

    Prints cluster sizes and mean fatality rate per cluster, plus
    key contextual shares.
    """
    print("\n--- K-means clustering with context features (illustrative) ---")

    feature_cols = [
        "year",
        "month",
        "day_of_week",
        "hour",
        "is_night",
        "is_peak",
        "is_weekend",
        "flag_teen",
        "flag_65plus",
        "flag_speed_related",
        "flag_impaired",
        "flag_hit_and_run",
        "flag_winter_road",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    if len(feature_cols) < 3:
        print("Not enough engineered features available for extended K-means.")
        return

    tmp = df.dropna(subset=feature_cols + ["is_fatal"]).copy()
    if tmp.empty:
        print("No rows available for K-means after dropping missing values.")
        return

    X = tmp[feature_cols]

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    tmp["cluster"] = km.fit_predict(X)

    print(f"Used features: {feature_cols}")
    summary = (
        tmp.groupby("cluster")
        .agg(
            count=("cluster", "size"),
            fatal_mean=("is_fatal", "mean"),
            night_share=("is_night", "mean"),
            speed_share=("flag_speed_related", "mean"),
            impaired_share=("flag_impaired", "mean"),
        )
        .reset_index()
    )
    for col in ["fatal_mean", "night_share", "speed_share", "impaired_share"]:
        summary[col] = summary[col].round(4)

    print(summary)


# -----------------------------------------------------------------
# Part 10. Visualizations (Saved to outputs/)
# -----------------------------------------------------------------


def _save_fig(outdir: Path, filename: str) -> None:
    """Save current matplotlib figure as a high-res PNG and close it."""
    outpath = outdir / filename
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {outpath}")


def plot_yearly_crashes_statewide(clean_statewide: pd.DataFrame, outdir: Path) -> None:
    """Figure 1: Yearly crash counts (Statewide)."""
    tbl = yearly_counts(clean_statewide)
    plt.figure()
    plt.plot(tbl["year"], tbl["crash_count"])
    plt.title("Statewide Pedestrian Crashes by Year (2010–2024)")
    plt.xlabel("Year")
    plt.ylabel("Number of Pedestrian Crashes")
    _save_fig(outdir, "fig01_yearly_crashes_statewide.png")


def plot_yearly_crashes_by_region(clean_dict: dict, outdir: Path) -> None:
    """Figure 2: Yearly crash counts by region (multiple lines)."""
    plt.figure()
    for region, df in clean_dict.items():
        tbl = yearly_counts(df)
        if not tbl.empty:
            plt.plot(tbl["year"], tbl["crash_count"], label=region)
    plt.title("Pedestrian Crashes by Year and Region (2010–2024)")
    plt.xlabel("Year")
    plt.ylabel("Number of Pedestrian Crashes")
    plt.legend()
    _save_fig(outdir, "fig02_yearly_crashes_by_region.png")


def plot_driver_age_rates(clean_statewide: pd.DataFrame, outdir: Path) -> None:
    """Figure 3: Teen vs 65+ driver involvement rates by year."""
    rates = minimal_flag_rollups(clean_statewide)
    plt.figure()
    plt.plot(rates["year"], rates["teen_rate"], label="Teen driver involved")
    plt.plot(rates["year"], rates["older65_rate"], label="65+ driver involved")
    plt.title("Teen vs. 65+ Driver Involvement Rates (Statewide, 2010–2024)")
    plt.xlabel("Year")
    plt.ylabel("Share of Crashes")
    plt.legend()
    _save_fig(outdir, "fig03_teen_vs_65plus_rates.png")


def plot_severity_distribution(clean_statewide: pd.DataFrame, outdir: Path) -> None:
    """Figure 4: Distribution of injury severity (ordinal)."""
    if "severity_ord" not in clean_statewide.columns:
        print("severity_ord not available; skipping severity distribution plot.")
        return

    s = clean_statewide["severity_ord"].dropna().astype(int)
    counts = s.value_counts().sort_index()

    labels = {
        0: "O (PDO)",
        1: "C (Possible)",
        2: "B (Non-incap.)",
        3: "A (Incap.)",
        4: "K (Fatal)",
    }
    x = list(counts.index)
    xlabels = [labels.get(i, str(i)) for i in x]

    plt.figure()
    plt.bar(range(len(x)), counts.values)
    plt.title("Distribution of Injury Severity (Statewide, 2010–2024)")
    plt.xlabel("Severity Level")
    plt.ylabel("Number of Crashes")
    plt.xticks(range(len(x)), xlabels, rotation=20, ha="right")
    _save_fig(outdir, "fig04_severity_distribution.png")


def plot_model_comparison(model_results: pd.DataFrame, outdir: Path) -> None:
    """Figure 5: Model comparison on is_fatal (Recall and F1)."""
    if model_results is None or model_results.empty:
        print("No model_results available; skipping model comparison plot.")
        return

    df = model_results.copy()
    df = df.set_index("model_name")[["recall", "f1"]].sort_values("recall", ascending=False)

    plt.figure()
    x = np.arange(len(df.index))
    width = 0.35
    plt.bar(x - width / 2, df["recall"].values, width, label="Recall (Fatal=1)")
    plt.bar(x + width / 2, df["f1"].values, width, label="F1 (Fatal=1)")
    plt.title("Model Performance Comparison on Fatality Prediction")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(x, df.index, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    _save_fig(outdir, "fig05_model_comparison_recall_f1.png")


def plot_random_forest_feature_importance(clean_statewide: pd.DataFrame, outdir: Path) -> None:
    """Figure 6: Random Forest feature importance for is_fatal prediction."""
    try:
        X, y = build_model_dataset(clean_statewide, use_context_flags=True, restrict_to_fatal_binary=True)
    except ValueError as e:
        print(f"Cannot build dataset for feature importance: {e}")
        return

    y = y.astype(int)
    if y.nunique() < 2:
        print("Target has only one class; skipping feature importance plot.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    importances = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(12)
    )

    plt.figure()
    plt.barh(importances.index[::-1], importances.values[::-1])
    plt.title("Top Feature Importances (Random Forest, Fatality Prediction)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    _save_fig(outdir, "fig06_rf_feature_importance.png")


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
    - run additional classifiers and clustering analyses
    - generate 6 visualizations and save to outputs/
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
    rates.to_csv(outdir / "statewide_teen_vs_65plus_rates.csv", index=False)

    # Part 5 — summary statistics
    for name, df_region in clean.items():
        quick_describe(name, df_region)

    # Part 6 — numerical summaries
    print_yearly_summaries(yr)

    # Part 7 — combine datasets into one ML-ready table
    combined = combine_clean_datasets(clean)
    print(f"\nCombined dataset shape (all regions, 2010–2024): {combined.shape}")

    # Part 8 — original ML examples (using statewide data)
    run_kmeans_example(statewide, n_clusters=4)
    run_logistic_regression_example(statewide)
    run_decision_tree_example(statewide, max_depth=5)

    # Part 9 — expanded ML analyses

    # 9.1 Model comparison for binary fatal vs non-fatal (with extra features)
    model_results = compare_classifiers_on_fatality(statewide)
    if not model_results.empty:
        model_results.to_csv(outdir / "model_comparison_is_fatal.csv", index=False)

    # 9.2 Multiclass severity model (0–4)
    run_multiclass_severity_tree(statewide, max_depth=6)

    # 9.3 Richer clustering with context features
    run_kmeans_with_context(statewide, n_clusters=5)

    # Part 10 — Visualizations (6 figures saved to outputs/)
    plot_yearly_crashes_statewide(statewide, outdir)
    plot_yearly_crashes_by_region(clean, outdir)
    plot_driver_age_rates(statewide, outdir)
    plot_severity_distribution(statewide, outdir)
    plot_model_comparison(model_results, outdir)
    plot_random_forest_feature_importance(statewide, outdir)

    print(f"\nSaved outputs (CSVs + figures) to: {outdir}")
