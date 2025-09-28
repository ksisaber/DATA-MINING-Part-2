# Data Mining Project – Part 2

This part of the project focuses on applying **Regression** and **Clustering** algorithms to the preprocessed climate dataset (Algeria). The work involves implementing, testing, and comparing multiple machine learning models, with an interactive interface for visualization and experimentation.

---

## Objectives

### Regression Tasks

* **Target Variable**: *Near-surface specific humidity*
* **Algorithms Implemented**:

  * Decision Trees (DT)
  * Random Forests (RF)
* **Steps**:

  1. Split dataset into training and testing sets
  2. Train DT and RF models with different parameters
  3. Evaluate and compare the models using:

     * Prediction accuracy metrics
     * Average execution time
  4. Compare custom implementations with library versions (e.g., scikit-learn)
  5. Save trained models for reuse (`tree_model_Summer.joblib`, `tree_model_Winter.joblib`)

### Clustering Tasks

* **Algorithms Implemented**:

  * CLARANS
  * DBSCAN
* **Steps**:

  1. Run clustering with different parameters
  2. Evaluate clustering quality with appropriate metrics
  3. Compare CLARANS and DBSCAN performances
  4. Visualize clusters with PCA for dimensionality reduction

---

## Advanced UI Features

* Select the Data Mining method to execute (Regression or Clustering)
* Enter a new data instance and predict regression output
* Display PCA visualization of clustering results

---

## Project Structure

```
Autumn.csv                -> Seasonal dataset (Autumn)
Spring.csv                -> Seasonal dataset (Spring)
Summer.csv                -> Seasonal dataset (Summer)
Winter.csv                -> Seasonal dataset (Winter)
data.csv                  -> Complete dataset
target.csv                -> Target variable (Near-surface specific humidity)

karim.py                  -> Utility functions and classes (algorithms & helpers)
interface.py              -> Streamlit interface (Regression + Clustering)

part1.ipynb               -> Jupyter notebook (Regression experiments)
part2.ipynb               -> Jupyter notebook (Clustering experiments)

tree_model_Summer.joblib  -> Saved Decision Tree model (Summer dataset)
tree_model_Winter.joblib  -> Saved Decision Tree model (Winter dataset)
```

---

## Requirements

* Python 3.8+
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

---

## How to Run

### Option 1: Jupyter Notebook

Explore experiments directly:

```bash
jupyter notebook part1.ipynb
jupyter notebook part2.ipynb
```

### Option 2: Streamlit Interface

Run the interactive app:

```bash
streamlit run interface.py
```

* Choose **Regression** or **Clustering** from the interface
* Adjust parameters and visualize results
* Test regression on a new input instance
* Display PCA visualization for clustering

---

## Outputs

* Trained Decision Tree and Random Forest models
* Regression performance comparison (accuracy, execution time)
* Clustering results with evaluation metrics
* Visualizations (scatter plots, PCA plots, decision boundaries)
* Saved models (`.joblib`) for future use
