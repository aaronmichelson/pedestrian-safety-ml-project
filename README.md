# Pedestrian Safety ML Project

This project analyzes pedestrian crash data from Wisconsin using basic preprocessing and simple machine learning examples.  
It supports Homework #6 (Milestone Report) for COMPSCI 715: Programming for Machine Learning.

## Files
- **pedestrian_safety_ml_project.py** — Main Python script  
- **datasets/** — Folder containing the 7 pedestrian crash CSV files  
- **outputs/** — Folder where yearly summaries and other results are saved  
- **README.md** — This file

## What the Python Script Does
1. Loads all 7 datasets  
2. Standardizes column names and parses crash dates  
3. Creates useful features (year, month, weekend flag, driver-age flags, fatality indicator)  
4. Filters data to 2010–2024  
5. Generates yearly crash counts for each region  
6. Combines all regions into a single dataset  
7. Runs basic machine learning examples:
   - K-means clustering  
   - Logistic regression  
   - Decision tree classifier  

## How to Run
Activate your Conda environment:

```
conda activate cs715env
```

Run the script:

```
python pedestrian_safety_ml_project.py
```

## Dependencies
- pandas  
- numpy  
- scikit-learn  

(Install using `conda install pandas numpy scikit-learn`)

## Notes
This README is intentionally simple.  
More details, figures, and explanations will be added in the final project.
