# Streaming Service Customer Analytics — Supervised & Unsupervised Learning


-----

## What this project does

This notebook applies supervised and unsupervised machine learning to a streaming service customer dataset. The work covers three separate tasks: predicting how much customers spend per month (regression), predicting whether a customer will churn (classification), and grouping customers into segments (clustering).

The goal wasn’t to find the most complex model. It was to compare approaches honestly and let the metrics decide.

-----

## Dataset

**File:** `Streaming.csv`  
**Size:** 5,000 customer records, 12 features  
**Targets:** `Monthly_Spend` (regression), `Churned` (classification)

Key features include Subscription Length, Satisfaction Score, Age, Discount Offered, Last Activity, Support Tickets Raised, and categorical variables like Region, Gender, and Payment Method.

Pre-processing steps applied:

- Removed `Customer_ID` (no predictive value)
- Median imputation for missing `Age` and `Satisfaction_Score` values (~10% missing each)
- StandardScaler on numerical features
- One-hot encoding on categorical features

-----

## Models implemented

**Regression (predicting Monthly Spend)**

|Model                                           |MAE |RMSE|R²   |
|------------------------------------------------|----|----|-----|
|Multiple Linear Regression                      |1.36|3.21|0.886|
|ANN (all features)                              |1.40|3.23|0.884|
|Random Forest Regressor                         |1.81|3.46|0.868|
|Best single-feature (Satisfaction Score, Linear)|5.16|6.49|0.534|

Multiple Linear Regression won. The data is largely linear, so throwing a neural network at it didn’t help.

**Classification (predicting churn)**

|Model                   |Accuracy|F1   |ROC-AUC|
|------------------------|--------|-----|-------|
|Random Forest Classifier|0.980   |0.977|0.989  |
|Logistic Regression     |0.806   |0.781|0.893  |

Churn patterns are non-linear; Random Forest picked them up where Logistic Regression couldn’t.

**Clustering (customer segmentation)**

|Algorithm                      |k|Silhouette|Davies-Bouldin|
|-------------------------------|-|----------|--------------|
|Hierarchical (complete linkage)|2|0.626     |0.573         |
|k-Means                        |2|0.230     |1.634         |

Both algorithms found k=2 as optimal. Hierarchical clustering scored better on every metric, though k-Means produced a more visually balanced split.

-----

## Key findings

Satisfaction Score and Subscription Length drive monthly spend. Everything else adds noise. A simple regression model with just the numerical features explains ~89% of variance in spend — the ANN and Random Forest didn’t close that remaining 11%.

For churn, the story flips. Random Forest’s ability to capture feature interactions matters a lot here, giving near-perfect precision and a 17-point accuracy gap over Logistic Regression.

-----

## How to run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/streaming-service-analytics.git
cd streaming-service-analytics

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook streaming_service.ipynb
```

Make sure `Streaming.csv` is in the same directory as the notebook before running.

-----

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
jupyter
```

-----

## Project structure

```
streaming-service-analytics/
│
├── streaming_service.ipynb   # Main notebook
├── Streaming.csv             # Dataset (add this before running)
├── requirements.txt
└── README.md
```

-----

## References

- Breiman, L. (2001) Random Forests. *Machine Learning*, 45(1), pp. 5–32.
- Goodfellow, I., Bengio, Y. and Courville, A. (2016) *Deep Learning*. MIT Press.
- James, G. et al. (2021) *An Introduction to Statistical Learning*. 2nd edn. Springer.
- MacQueen, J. (1967) Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium*, pp. 281–297.
- Rousseeuw, P.J. (1987) Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, pp. 53–65.