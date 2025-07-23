# NO2 Power Price Forecasting

This project aims to predict **day-ahead electricity prices** for the NO2 zone in Norway using a diverse set of energy-related features. It leverages data from the ENTSO-E API, incorporating load forecasts, renewable generation, net transfer capacities, and more.

---

## Features Used

- **Day-Ahead Electricity Prices** (target variable)
- **Load Forecasts** (NO2 + neighboring zones)
- **Wind & Solar Forecasts**
- **Generation Forecasts**
- **Net Transfer Capacities (NTC)** across borders
- **Physical Cross-Border Flows**

---

## Tools & Technologies

- Python (Pandas, NumPy, Scikit-learn, Seaborn)
- `entsoe-py` for ENTSO-E API access
- Linear Regression, Decision Tree, Random Forest, XGBoost, SVM for prediction
- Jupyter Notebook for analysis

---


### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/no2-power-price-forecasting.git
cd no2-power-price-forecasting

# Install dependencies
pip install -r requirements.txt

