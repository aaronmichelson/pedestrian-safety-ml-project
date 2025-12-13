# Pedestrian Safety Machine Learning Pipeline (Wisconsin, 2010–2024)

This project develops an interpretable and reproducible machine learning (ML)
pipeline to analyze pedestrian crash data from Wisconsin between 2010 and 2024.
The analysis integrates statewide, regional, and Milwaukee County datasets to
explore temporal trends, demographic patterns, crash severity, and the
challenges of predicting rare fatal outcomes.

Rather than optimizing for maximum predictive accuracy, the project emphasizes
**transparent preprocessing**, **feature engineering**, **class imbalance
handling**, and **model interpretability**, consistent with contemporary
transportation safety research.

---

## Project Structure

```
FinalProject/
│
├── DataSets/                      # Input datasets (7 CSVs; local, not tracked)
│   ├── PedestrianCrashes_Wisconsin_2001-2024.csv
│   ├── PedestrianCrashes_Wisconsin_SWRegion_2001-2024.csv
│   ├── PedestrianCrashes_Wisconsin_SERegion_2001-2024.csv
│   ├── PedestrianCrashes_Wisconsin_NERegion_2001-2024.csv
│   ├── PedestrianCrashes_Wisconsin_NCRegion_2001-2024.csv
│   ├── PedestrianCrashes_Wisconsin_NWRegion_2001-2024.csv
│   └── PedestrianCrashes_Wisconsin_MilwaukeeCounty_2001-2024.csv
│
├── PythonFiles/
│   └── pedestrian_safety_ml_project.py   # Main pipeline script
│
├── outputs/                       # Generated tables and figures (auto-created)
│   ├── yearly_*.csv
│   ├── model_comparison_is_fatal.csv
│   ├── fig01_yearly_crashes_statewide.png
│   ├── fig02_yearly_crashes_by_region.png
│   ├── fig03_teen_vs_65plus_rates.png
│   ├── fig04_severity_distribution.png
│   ├── fig05_model_comparison_recall_f1.png
│   └── fig06_rf_feature_importance.png
│
└── README.md
```

---

## What the Pipeline Does

The main Python script executes a full end-to-end workflow:

### 1. Data Ingestion
- Loads **seven pedestrian crash datasets** (statewide, five WisDOT regions,
  and Milwaukee County)
- Stores datasets in a dictionary keyed by region name

### 2. Standardized Preprocessing
- Normalizes column names across datasets
- Identifies and parses crash date fields
- Filters records to **2010–2024** to avoid structural severity imbalance

### 3. Feature Engineering
- Temporal features: year, month, day of week, hour, night/peak indicators
- Driver demographics: teen and 65+ involvement flags
- Severity measures: binary fatality and ordinal injury severity
- Crash context flags (when available): speed, impairment, hit-and-run,
  winter conditions, vehicle type, construction zones

### 4. Aggregation and Descriptive Analysis
- Yearly crash counts by region
- Driver age involvement trends
- Descriptive statistics for engineered variables

### 5. Machine Learning Analyses
- K-means clustering (basic and extended feature sets)
- Logistic regression (baseline and class-weighted)
- Decision tree classifier
- Random Forest and Gradient Boosting classifiers
- Multiclass injury severity modeling

### 6. Model Evaluation and Interpretation
- Recall-, precision-, and F1-oriented evaluation
- Threshold tuning for rare fatal outcomes
- Permutation-based feature importance

### 7. Visualization Outputs
- Statewide and regional crash trends
- Injury severity distribution
- Model performance comparisons
- Feature importance rankings

All figures and tables are generated programmatically and saved for direct
inclusion in the final report.

---

## How to Run

1. Activate your Conda environment:

```bash
conda activate cs715env
```

2. Navigate to the project root:

```bash
cd path/to/FinalProject
```

3. Run the pipeline:

```bash
python PythonFiles/pedestrian_safety_ml_project.py
```

All outputs will be written to:

```
FinalProject/outputs/
```

---

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- pathlib (standard library)

Install missing packages with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Notes

- Raw datasets are **not included** in the repository.
- Outputs are fully reproducible by rerunning the script.
- Predictive performance is constrained primarily by feature availability.
- The pipeline is designed for future extension with roadway, spatial, and
  environmental data.

---

## Author

**Aaron Michelson**  
Final Project — Programming for Machine Learning  
University of Wisconsin–Milwaukee
