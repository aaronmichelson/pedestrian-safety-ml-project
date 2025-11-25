# Pedestrian Safety ML Project

This project analyzes pedestrian crash data from Wisconsin (2010--2024)
using a custom preprocessing pipeline and several introductory machine
learning models. The goal is to explore regional crash patterns,
construct meaningful features, and generate early insights that will
support a larger predictive and analytical study.

## Project Structure

    FinalProject/
    │
    ├── DataSets/              # Input datasets (7 CSVs: statewide + 5 regions + Milwaukee)
    ├── outputs/               # All generated results, summaries, and model outputs
    ├── pedestrian_safety_ml_project.py   # Main Python script (full pipeline)
    └── README.md              # Project documentation

## What the Python Pipeline Does

The main script performs the following steps:

1.  **Loads all datasets** (Statewide, SW, SE, NE, NC, NW, Milwaukee
    County)
2.  **Standardizes** column names and parses crash dates
3.  **Builds derived features**, including:
    -   `year`, `month`, `day_of_week`
    -   `is_weekend`
    -   `flag_teen` (teen driver indicator)
    -   `flag_65plus` (senior driver indicator)
    -   `is_fatal` (fatal injury severity)
4.  **Filters records** to the consistent reporting window (2010--2024)
5.  **Computes yearly crash counts** for each region and saves them to
    `outputs/`
6.  **Generates additional summaries**, such as teen vs. 65+ driver
    involvement
7.  **Combines all regions** into a single integrated dataset for ML
    analysis
8.  **Runs simple baseline ML models**, including:
    -   K-means clustering\
    -   Logistic regression\
    -   Decision tree classifier

These models are meant as illustrative baselines and starting points for
deeper analysis.

## How to Run

1.  Activate your Conda environment:

``` bash
conda activate cs715env
```

2.  Navigate to the project directory:

``` bash
cd path/to/FinalProject
```

3.  Run the pipeline:

``` bash
python pedestrian_safety_ml_project.py
```

All results will be saved automatically into:

    FinalProject/outputs/

## Dependencies

This project uses the following Python libraries:

-   pandas\
-   numpy\
-   scikit-learn\
-   pathlib (standard library)

To install missing packages:

``` bash
pip install pandas numpy scikit-learn
```

## Notes

-   The datasets and all outputs are stored locally and are **not
    included in the repository**.
-   Additional visualizations, modeling techniques, and statistical
    analysis will be added as the project develops.
-   This README is written to be project-ready and independent of any
    coursework references.
