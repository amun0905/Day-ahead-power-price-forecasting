# Forecasting Day-ahead Electricity Prices in the NO2 Bidding Zone

This repository contains the code for forecasting NO2 Day-ahead Power Prices using ENTSO-e Data and traditional machine learning models. 

---

## Project Overview

This project does the following:

- Retrieves publicly available data from the ENTSO-e API.
- Engineers relevant supply–demand features.
- Trains multiple regression and ML models to forecast day-ahead electricity prices.
- Evaluates and compare model performance using statistical metrics.

---

## Data Sources

All data were retrieved via the ENTSO-e Transparency Platform API, using the Pandas client.

**Features used:**
- Day-ahead prices (target variable)
- Load forecasts (NO2 and neighbouring zones)
- Generation forecasts
- Renewable generation (wind, solar)
- Net Transfer Capacities (NTC)
- Cross-border physical flows
- Aggregate water reservoir levels

**Feature engineering included:**
- Lag features (1, 7, 30 days)
- Rolling means and standard deviations
- Interaction terms (e.g., load–generation ratios)
- Temporal indicators (day of week, month, weekend)

---

## Models Implemented

| Model | Description | Library |
|--------|--------------|----------|
| Linear Regression | Baseline statistical model | `scikit-learn` |
| Decision Tree | Nonlinear regression with tree-based splits | `scikit-learn` |
| Random Forest | Ensemble of decision trees using bagging | `scikit-learn` |
| XGBoost | Gradient boosting ensemble | `xgboost` |
| Support Vector Machine (SVM) | Regression with kernel-based feature space | `scikit-learn` |

---

## Model Performance

| Model | MSE | MAE | R² | Adjusted R² |
|--------|-----|-----|----|--------------|
| Linear Regression | 17.88 | 2.59 | 0.85 | 0.84 |
| Decision Tree | 24.25 | 3.06 | 0.80 | 0.79 |
| Random Forest | 16.96 | 2.33 | 0.86 | 0.85 |
| XGBoost | 16.33 | 2.62 | 0.86 | 0.85 |
| SVM | 15.19 | 2.18 | 0.87 | 0.86 |

> Best performing model: Support Vector Machine (SVM)  
> Balanced accuracy, lowest error, and strong generalization to unseen data.

---

## Key Insights

- **Prev_Day_DA_prices_NO_2** was the most influential feature in tree-based models.  
- SVM and XGBoost distributed feature importance more evenly, improving robustness.  
- All models struggled slightly with extreme price spikes.

