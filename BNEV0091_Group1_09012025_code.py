#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# This project focuses on predicting day-ahead electricity prices for the NO2 zone in Norway. The analysis incorporates various energy-related data, such as forecasts for load, generation, wind, and solar energy, as well as cross-border physical flows and net transfer capacities. The goal is to build a comprehensive dataset that can be used for predictive modeling.

# # **Data Collection**
# Data is collected using the **ENTSO-E API**.
# 
# The following datasets are retrieved:
# 
# 1.   **Day-Ahead Prices**: Historical day-ahead electricity prices for the NO2 zone.
# 2.   **Load Forecasts**: Forecasted electricity demand for multiple areas, including NO2 and neighboring regions.
# 3.   **Wind and Solar Forecasts**: Forecasted renewable energy production (wind and solar) for relevant zones.
# 1.   **Generation Forecasts**: Predicted electricity generation capacity for selected regions.
# 1.   **Net Transfer Capacities (NTC)**: Week-ahead net transfer capacities for specific cross-border connections.
# 1.   **Cross-Border Physical Flows**: Net flows of electricity across borders involving the NO2 zone.
# 
# The analysis spans from October 2023 to September 2024. To account for lagged features, the dataset includes an extended start date.

# In[1]:


#!{sys.executable} -m pip install entsoe-py
get_ipython().system('pip install entsoe-py')


# 

# In[1]:


# 1. Imports and API Data Collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from entsoe import EntsoePandasClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats

# Initialize API Client
client = EntsoePandasClient(api_key='d43c0033-144a-4c29-aa12-98d6d1070332')

# Define date range
start = pd.Timestamp('20231001', tz='Europe/Brussels')
end = pd.Timestamp('20240930', tz='Europe/Brussels')
extended_start = start - pd.Timedelta(days=1)

# Define specific areas and variable mappings
areas_load_forecast = ["NO_2", "NO_1", "NO_5", "DK", "NL", "DE_LU"]
areas_wind_and_solar_forecast = areas_load_forecast  # Same as load forecast areas
areas_generation_forecast = areas_load_forecast  # Same as load forecast areas
ntc_pairs = [
    ("NO_2", "NL"),
    ("NO_2", "DK1"),
    ("NO_2", "DE_LU"),
    ("NO_2", "GB"),
    ("NO_2", "NO_5")
]  # Net transfer capacity

# Cross-Border Physical Flows for NO2 Zone
crossborder_pairs = ntc_pairs  # Use the existing NTC pairs for cross-border flow calculations

data_frames = {}

# Fetch Day-Ahead Prices for NO_2
try:
    data_frames['DA_prices_NO_2'] = client.query_day_ahead_prices("NO_2", start=extended_start, end=end).to_frame(name='DA_prices_NO_2')
    # Add Previous Day's Price as a Feature
    data_frames['DA_prices_NO_2']['Prev_Day_DA_prices_NO_2'] = data_frames['DA_prices_NO_2']['DA_prices_NO_2'].shift(1)
    # Filter to remove the extended day
    data_frames['DA_prices_NO_2'] = data_frames['DA_prices_NO_2'].loc[start:end]
except Exception as e:
    print(f"Failed to fetch DA prices for NO_2: {e}")

# Fetch Load Forecast for multiple areas
for area in areas_load_forecast:
    try:
        data = client.query_load_forecast(area, start=start, end=end).rename(columns={'Forecasted Load': f'Load_forecast_{area}'})
        data_frames[f'Load_forecast_{area}'] = data
    except Exception as e:
        print(f"Failed to fetch Load Forecast for {area}: {e}")

# Fetch Wind and Solar Forecast for multiple areas
for area in areas_wind_and_solar_forecast:
    try:
        data = client.query_wind_and_solar_forecast(area, start=start, end=end)
        data.columns = [f"{col}_{area}" for col in data.columns]  # Consistent column naming
        data_frames[f'Wind_and_Solar_forecast_{area}'] = data
    except Exception as e:
        print(f"Failed to fetch Wind and Solar Forecast for {area}: {e}")

# Fetch Generation Forecast for multiple areas
for area in areas_generation_forecast:
    try:
        data = client.query_generation_forecast(area, start=start, end=end).to_frame(name=f'Generation_forecast_{area}')
        data_frames[f'Generation_forecast_{area}'] = data
    except Exception as e:
        print(f"Failed to fetch Generation Forecast for {area}: {e}")

# Fetch Net Transfer Capacity (Week-Ahead) for specified pairs
for from_area, to_area in ntc_pairs:
    try:
        data = client.query_net_transfer_capacity_weekahead(from_area, to_area, start=start, end=end).to_frame(
            name=f'NTC_WeekAhead_{from_area}_to_{to_area}')
        data_frames[f'NTC_WeekAhead_{from_area}_to_{to_area}'] = data
    except Exception as e:
        print(f"Failed to fetch NTC Week-Ahead from {from_area} to {to_area}: {e}")

# Fetch Aggregate Water Reservoirs and Hydro Storage for NO_2
try:
    data_frames['Aggregate_Water_Reservoirs_NO_2'] = client.query_aggregate_water_reservoirs_and_hydro_storage(
        "NO_2", start=start, end=end).to_frame(name='Aggregate_Water_Reservoirs_NO_2')
except Exception as e:
    print(f"Failed to fetch Aggregate Water Reservoirs for NO_2: {e}")

# Fetch and compute net flows for each pair
def fetch_net_flow(from_area, to_area):
    try:
        flow_1 = client.query_crossborder_flows(from_area, to_area, start=start, end=end)
        flow_2 = client.query_crossborder_flows(to_area, from_area, start=start, end=end)
        net_flow = flow_1 - flow_2
        net_flow.name = f"Net_Flow_{from_area}_to_{to_area}"
        return net_flow.to_frame()
    except Exception as e:
        print(f"Failed to fetch cross-border flows between {from_area} and {to_area}: {e}")
        return pd.DataFrame()

for from_area, to_area in crossborder_pairs:
    data_frames[f"Net_Flow_{from_area}_to_{to_area}"] = fetch_net_flow(from_area, to_area)


# All datasets are merged into a single DataFrame for analysis, ensuring all datasets align on the same timestamps.

# In[3]:


merged_data = pd.concat(data_frames.values(), axis=1)

merged_data.info()
print(merged_data.describe())
print(merged_data.head(10))


# Missing values are addressed using forward-fill and backward-fill techniques where appropriate.

# In[4]:


# Filter rows where 'DA_prices_NO_2' is not NaN
merged_data = merged_data.dropna(subset=['DA_prices_NO_2'])

# Check the result
print(merged_data.info())


merged_data = merged_data.ffill()
merged_data = merged_data.bfill()
print("After fill the data")
print(merged_data.info())

print(merged_data.head(10))


# # **Exploratory Data Analysis (EDA)**
# 
# EDA is conducted to understand the distributions, relationships, and variability of the collected data:
# - **Target Variable**: The distribution of day-ahead prices (`DA_prices_NO_2`) is analyzed for trends and seasonality.
# - **Feature Distributions**: Boxplots and histograms reveal variability and outliers in key features.
# - **Correlation Analysis**: A heatmap highlights relationships between features and the target variable.
# - **Temporal Patterns**: Trends in electricity prices and other features over time are visualized.
# 

# In[5]:


# Plot time-series of DA_prices_NO_2
merged_data["DA_prices_NO_2"].plot(figsize=(12, 6))
plt.title("Time Series of DA_prices_NO_2")
plt.xlabel("Date")
plt.ylabel("Price (EUR/MWh)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()


# Visualize the distribution of the target variable
sns.histplot(merged_data["DA_prices_NO_2"], kde=True, color='blue')
plt.title('Distribution of DA_prices_NO_2')
plt.xlabel('Price (EUR/MWh)')
plt.ylabel('Frequency')
plt.show()

# Calculate mean and standard deviation of the target variable
mean_price = merged_data["DA_prices_NO_2"].mean()
std_price = merged_data["DA_prices_NO_2"].std()
print(f"Mean Price: {mean_price:.2f} EUR/MWh, Standard Deviation: {std_price:.2f} EUR/MWh")


# In[6]:


# Visualize the features

# Determine the number of columns in merged_data
num_plots = len(merged_data.columns)

# Dynamically calculate rows and columns
num_cols = 4
num_rows = (num_plots + num_cols - 1) // num_cols

# Create the figure with adjusted size
plt.figure(figsize=(15, num_rows * 5))

# Plot a box plot for each column in merged_data
for i, column in enumerate(merged_data.columns, start=1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(y=merged_data[column], color=np.random.rand(3,))
    plt.title(f'Box Plot of {column}', fontsize=10)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()  # Prevent overlap
plt.show()


# In[7]:


# Identify features with all zero values
all_zero_features = [col for col in merged_data.columns if (merged_data[col] == 0).all()]

# Identify features with 25%, 50%, and 75% values being the same (low variability)
low_variability_features = [
    col for col in merged_data.columns
    if merged_data[col].describe()[['25%', '50%', '75%']].nunique() == 1
]

# Combine both lists of features to drop
features_to_drop = set(all_zero_features + low_variability_features)

# Drop these features from the DataFrame
merged_data = merged_data.drop(columns=features_to_drop)

print(f"Removed features: {features_to_drop}")


# In[8]:


print(merged_data.isna().sum())
merged_data.describe()


# In[9]:


# Outlier Detection
# Calculate IQR for each numeric column
Q1 = merged_data.quantile(0.25)  # First quartile (25th percentile)
Q3 = merged_data.quantile(0.75)  # Third quartile (75th percentile)
IQR = Q3 - Q1  # Interquartile range

# Define the outlier threshold
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out rows where any column's value is an outlier
merged_data_no_outliers = merged_data[~((merged_data < lower_bound) | (merged_data > upper_bound)).any(axis=1)]




# Data Visualization: Box Plots After Removing Outliers

plt.figure(figsize=(15, num_rows * 5))

# Plot a box plot for each column in merged_data_no_outliers (after removing outliers)
for i, column in enumerate(merged_data_no_outliers.columns, start=1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(y=merged_data_no_outliers[column], color=np.random.rand(3,))
    plt.title(f'Box Plot of {column}', fontsize=10)
    plt.xlabel('')
    plt.ylabel('')


plt.tight_layout()
plt.show()


# In[10]:


print("After removing outliers:")
print(merged_data_no_outliers.info())


# In[11]:


# Add a 'Weekday' column (0=Monday, 6=Sunday) and a 'Weekend' column (1=Weekend, 0=Weekday)
merged_data_no_outliers['Weekday'] = merged_data_no_outliers.index.dayofweek
merged_data_no_outliers['Is_Weekend'] = (merged_data_no_outliers['Weekday'] >= 5).astype(int)

# Load forecasts on weekdays vs weekends
plt.figure(figsize=(6, 4))
sns.boxplot(data=merged_data_no_outliers, x='Is_Weekend', y='Load_forecast_NO_2', palette='coolwarm')
plt.xticks([0, 1], ['Weekday', 'Weekend'])
plt.title('Load Forecast Variability: Weekday vs Weekend')
plt.xlabel('Day Type')
plt.ylabel('Load Forecast (MW)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# NTC values against day-ahead prices
plt.figure(figsize=(6, 4))
sns.scatterplot(data=merged_data_no_outliers, x='NTC_WeekAhead_NO_2_to_NL', y='DA_prices_NO_2', alpha=0.7)
plt.title('NTC vs Day-Ahead Prices')
plt.xlabel('Net Transfer Capacity (MW)')
plt.ylabel('Day-Ahead Prices (EUR/MWh)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Net flows against day-ahead prices
plt.figure(figsize=(6, 4))
sns.scatterplot(data=merged_data_no_outliers, x='Net_Flow_NO_2_to_GB', y='DA_prices_NO_2', alpha=0.7)
plt.title('Net Flow vs Day-Ahead Prices')
plt.xlabel('Net Flow (MW)')
plt.ylabel('Day-Ahead Prices (EUR/MWh)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[12]:


# Correlation Matrix Heatmap

threshold = 0.1

correlation_matrix = merged_data_no_outliers.corr()

# Filter out features with low correlation to the target or low maximum correlation overall
low_corr_features = [
    col for col in correlation_matrix.columns
    if (abs(correlation_matrix.loc['DA_prices_NO_2', col]) < threshold) or
       (correlation_matrix[col].abs().max() < threshold)
]

# Filter the correlation matrix to exclude low-correlation features
filtered_corr_matrix = correlation_matrix.drop(index=low_corr_features, columns=low_corr_features)

# Mask the upper triangle of the filtered correlation matrix
mask = np.triu(np.ones_like(filtered_corr_matrix, dtype=bool))

# Create the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm', cbar=True,
            mask=mask, annot_kws={"size": 8}, fmt='.2f', square=True)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0, ha='right')
plt.title('Filtered Correlation Matrix (Triangle Format)')

plt.tight_layout()
plt.show()



# In[13]:


# Generate pair plots for the entire DataFrame
sns.pairplot(merged_data_no_outliers)
plt.show()


# In[14]:


# Determine the number of columns
number_cols = merged_data_no_outliers.shape[1]

slice_size = 4

# Iterate over the dataset, slicing it into chunks of 4 columns
for start_col in range(0, number_cols, slice_size):

    end_col = min(start_col + slice_size, number_cols)
    data_slice = merged_data_no_outliers.iloc[:, start_col:end_col]


    print(f"Visualizing columns {start_col + 1} to {end_col}: {list(data_slice.columns)}")


    sns.pairplot(data_slice,plot_kws={'s': 10})
    plt.show()


# In[15]:


# Calculate the correlation matrix
correlation_matrix = merged_data_no_outliers.corr()

# Display correlations of all variables with the target variable
relevant_features = correlation_matrix['DA_prices_NO_2'].sort_values(ascending=False)
print("Correlation with DA_prices_NO2:\n", relevant_features)


# In[16]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Add a constant column to the data for VIF calculation (the intercept in a regression model)
X = add_constant(merged_data_no_outliers.drop('DA_prices_NO_2', axis=1))

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF for each feature
print(vif_data)


# # **Feature Engineering**
# 
# To enhance the predictive power of the dataset, the following features are engineered:
# 1. **Lagged Features**: Historical values for load and generation forecasts at intervals of 1, 7, and 30 days.
# 2. **Rolling Statistics**: Rolling means and standard deviations for load and generation forecasts over 3, 7, and 30 days.
# 3. **Relative Changes**: Percentage changes in load and generation forecasts.
# 4. **Interaction Terms**: Ratios and differences between load and generation values.
# 5. **Temporal Features**: Day of the week, month, and weekend indicators are added to capture seasonality and periodic effects.
# 
# These features are critical for capturing patterns in electricity demand, supply, and market dynamics.
# 

# In[17]:


merged_data_no_outliers = merged_data_no_outliers.copy()

# Create lagged features for load and generation forecasts
lags = [1, 7, 30]
for lag in lags:
    merged_data_no_outliers.loc[:, f'Load_forecast_NO_2_lag_{lag}'] = merged_data_no_outliers['Load_forecast_NO_2'].shift(lag)
    merged_data_no_outliers.loc[:, f'Generation_forecast_NO_2_lag_{lag}'] = merged_data_no_outliers['Generation_forecast_NO_2'].shift(lag)

# Add rolling statistics features (mean and SD)
window_sizes = [3, 7, 30]
for window in window_sizes:
    merged_data_no_outliers.loc[:, f'Load_forecast_NO_2_roll_mean_{window}'] = (
        merged_data_no_outliers['Load_forecast_NO_2'].rolling(window=window).mean()
    )
    merged_data_no_outliers.loc[:, f'Load_forecast_NO_2_roll_std_{window}'] = (
        merged_data_no_outliers['Load_forecast_NO_2'].rolling(window=window).std()
    )
    merged_data_no_outliers.loc[:, f'Generation_forecast_NO_2_roll_mean_{window}'] = (
        merged_data_no_outliers['Generation_forecast_NO_2'].rolling(window=window).mean()
    )
    merged_data_no_outliers.loc[:, f'Generation_forecast_NO_2_roll_std_{window}'] = (
        merged_data_no_outliers['Generation_forecast_NO_2'].rolling(window=window).std()
    )

# Add relative change features
merged_data_no_outliers.loc[:, 'Load_forecast_NO_2_pct_change'] = (
    merged_data_no_outliers['Load_forecast_NO_2'].pct_change()
)
merged_data_no_outliers.loc[:, 'Generation_forecast_NO_2_pct_change'] = (
    merged_data_no_outliers['Generation_forecast_NO_2'].pct_change()
)

# Add interaction features
merged_data_no_outliers.loc[:, 'Load_Generation_ratio'] = (
    merged_data_no_outliers['Load_forecast_NO_2'] / merged_data_no_outliers['Generation_forecast_NO_2']
)
merged_data_no_outliers.loc[:, 'Load_Generation_diff'] = (
    merged_data_no_outliers['Load_forecast_NO_2'] - merged_data_no_outliers['Generation_forecast_NO_2']
)

# Add datetime features
merged_data_no_outliers.loc[:, 'Day_of_Week'] = merged_data_no_outliers.index.dayofweek  # 0=Monday, 6=Sunday
merged_data_no_outliers.loc[:, 'Month'] = merged_data_no_outliers.index.month
merged_data_no_outliers.loc[:, 'Is_Weekend'] = (merged_data_no_outliers['Day_of_Week'] >= 5).astype(int)

# Drop rows with NaN values caused by lagging, rolling, and pct_change
merged_data_no_outliers.dropna(inplace=True)


print(merged_data_no_outliers.head())
print(merged_data_no_outliers.columns)


# In[18]:


# Ensure a proper DatetimeIndex
if not isinstance(merged_data_no_outliers.index, pd.DatetimeIndex):
    raise ValueError("The index of merged_data_no_outliers must be a DatetimeIndex for proper slicing.")

#Define X,y
X = merged_data_no_outliers.drop('DA_prices_NO_2', axis=1)
y = merged_data_no_outliers['DA_prices_NO_2']

# Train/Test Split
train_data = merged_data_no_outliers.loc['2023-10-01':'2024-06-30']
test_data = merged_data_no_outliers.loc['2024-07-01':'2024-09-30']

# Separate Predictors and Target
X_train, y_train = train_data.drop('DA_prices_NO_2', axis=1), train_data['DA_prices_NO_2']
X_test, y_test = test_data.drop('DA_prices_NO_2', axis=1), test_data['DA_prices_NO_2']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {y_train.shape}")
print(f"Y_test shape: {y_test.shape}")


# In[19]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify the scaling by checking mean and standard deviation of the training set
scaled_summary = {
    "Training Set Mean (Scaled)": X_train_scaled.mean(axis=0),
    "Training Set Std Dev (Scaled)": X_train_scaled.std(axis=0)
}

scaled_summary


# In[20]:


# Build the OLS model
import statsmodels.api as sm
X_train_with_constant = sm.add_constant(X_train_scaled)
X_test_with_constant = sm.add_constant(X_test_scaled)

# Fit in OLS model
ols_model = sm.OLS(y_train, X_train_with_constant).fit()

print(ols_model.summary())

# Prediction
y_pred_ols = ols_model.predict(X_test_with_constant)

# Evaluate
from sklearn.metrics import mean_squared_error
mse_ols = mean_squared_error(y_test, y_pred_ols)
print(f'Mean Squared Error (OLS): {mse_ols}')


# In[21]:


# Model Diagnostics
# Variance Inflation Factor (VIF) Analysis
# Import necessary library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)

vif_data = pd.DataFrame()
vif_data['Feature'] = X_train_df.columns
vif_data['VIF'] = [variance_inflation_factor(X_train_df.values, i) for i in range(X_train_df.shape[1])]
vif_data


# In[22]:


# Model Diagnostics
# Residuals vs Fitted Plot (Homoscedasticity Check)
# Residuals Analysis
residuals = ols_model.resid

# Plot residuals vs fitted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=ols_model.fittedvalues, y=residuals, color='blue', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()


# In[23]:


# Model Diagnostics
# Q-Q Plot (Normality Check)
# Q-Q plot of residuals
sm.qqplot(residuals, line="45", fit=True)
plt.title("Q-Q Plot of Residuals")
plt.show()


# In[24]:


# Model Diagnostics
# Shapiro-Wilk Test Results with Conditional Message

from scipy.stats import shapiro  # Import the Shapiro-Wilk test function

# Perform Shapiro-Wilk test
shapiro_test = shapiro(residuals)

# Print results with conditional interpretation
print("### Shapiro-Wilk Test Results ###")
print(f"Test Statistic: {shapiro_test.statistic:.4f}")
print(f"p-value: {shapiro_test.pvalue:.4e}")

# Conditional message
if shapiro_test.pvalue < 0.05:
    print("\nMessage: The residuals deviate significantly from normality. Consider applying transformations "
          "(e.g., log or square root) or exploring non-linear regression techniques.")
else:
    print("\nMessage: The residuals do not significantly deviate from normality. The normality assumption is satisfied.")


# In[25]:


# Model Diagnostics
# Leverage vs Residuals Analysis
# Leverage vs Residuals Plot
fig, ax = plt.subplots(figsize=(8, 6))
sm.graphics.plot_leverage_resid2(ols_model, ax=ax, color='orange')
plt.title('Leverage vs Residuals (Refined Log-Transformed Model)')
plt.grid(True)
plt.show()


# In[26]:


# Model Diagnostics
# Cook's Distance Plot
# Calculate Cook's Distance manually
influence = ols_model.get_influence()
cooks_d = influence.cooks_distance[0]

# Create the custom Cook's Distance plot with orange points
fig, ax = plt.subplots(figsize=(10, 6))
# Remove 'use_line_collection=True' since it is not available in older versions.
# Instead we plot the stem lines and markers with respective colors
markerline, stemlines, baseline = plt.stem(range(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ")
plt.setp(stemlines, 'color', 'orange')  # set stem lines color to orange
plt.setp(markerline, 'color', 'orange') # set marker line color to orange


plt.axhline(4 / len(X_train), color='red', linestyle='--', label="Threshold (4/n)")
plt.title("Cook's Distance Plot (Orange Color)")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.legend()
plt.grid(True)
plt.show()


# In[27]:


# Evaluate Test Set Performance
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
X_test_with_constant = sm.add_constant(X_test_scaled)
y_test_pred = ols_model.predict(X_test_with_constant)

# Calculate R^2 and RMSE for the test set
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Display evaluation metrics
test_performance = {
    "Test R^2": test_r2,
    "Test RMSE": test_rmse
}

test_performance


# # **Model Testing**
# 
# Testing differen Models and fine tuning them
# 1. **Simple Linear Regression**
# 2. **Decision Tree**
# 3. **Random Forest**
# 4. **XGBoost**
# 5. **SVM**

# # **Simple Linear Regression**

# In[35]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Predictions and Metrics
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - Mean Squared Error (MSE): {mse_lr}")
print(f"Linear Regression - Mean Absolute Error (MAE): {mae_lr}")
print(f"Linear Regression - R-squared (R²): {r2_lr}")

# Feature Importances (For Linear Regression, coefficients represent feature importance)
lr_feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lr_model.coef_
}).sort_values(by='Importance', ascending=False)

print("Top 10 Feature Importances for Linear Regression:")
print(lr_feature_importances.head(10))

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', linestyle='-', marker='o', color='blue')
plt.plot(y_test.index, y_pred_lr, label='Predicted Prices', linestyle='--', marker='x', color='orange')
plt.fill_between(y_test.index, y_pred_lr - mse_lr**0.5, y_pred_lr + mse_lr**0.5, color='green', alpha=0.2, label='Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Day-Ahead Price (EUR)')
plt.title('Actual vs Predicted Day-Ahead Prices with Linear Regression')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


# # **Decision Tree**

# **Fine Tuning**

# In[36]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#Define the base parameters and tune min_samples_split and min_samples_leaf
param_grid_1 = {
    'min_samples_split': [3, 4],  # Values to test for min_samples_split
    'min_samples_leaf': [6, 8, 10]  # Values to test for min_samples_leaf
}

base_params = {
    'max_depth': 30,
    'random_state': 42
}

#Perform Grid Search
grid_search_1 = GridSearchCV(
    estimator=DecisionTreeRegressor(**base_params),
    param_grid=param_grid_1,
    cv=3,  # Cross-validation folds
    verbose=2,  # Display progress during search
    n_jobs=-1  # Use all available processors
)

#Fit the grid search with training data
grid_search_1.fit(X_train, y_train)

#Get the best parameters and combine them with the base parameters
best_params_final = {**base_params, **grid_search_1.best_params_}

# Output best parameters
print(f"Best parameters after tuning min_samples_split and min_samples_leaf: {best_params_final}")


# In[37]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#Test same parameters more finely based on initial best parameters found previously for min_samples_split and min_samples_leaf
param_grid_2 = {
    'min_samples_split': [2, 3],
    'min_samples_leaf': [15, 20, 25] 
}

#Initialise the Decision Tree with the base parameters
base_params = {
    'max_depth': 30,  # Example of a parameter from Step 1
    'random_state': 42
}

#Perform Grid Search
grid_search_2 = GridSearchCV(
    estimator=DecisionTreeRegressor(**base_params),
    param_grid=param_grid_2,
    cv=3,  # Cross-validation folds
    verbose=2,  # Display progress during search
    n_jobs=-1  # Use all available processors
)

#Fit the grid search with training data
grid_search_2.fit(X_train, y_train)

#Get the best parameters and combine them with the base parameters
best_params_final = {**base_params, **grid_search_2.best_params_}

#Output best parameters
print(f"Best parameters after tuning min_samples_split and min_samples_leaf: {best_params_final}")


# In[38]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#Define the max_depth values to test
param_grid_3 = {
    'max_depth': [5, 10, 15, 20, 25, 30, None]
}

#Initialise the Decision Tree with the base parameters and teh bezt valuse found for min_samples_leaf and min_samples_split
base_params = {
    'max_depth': 30, 
    'random_state': 42,
    "min_samples_leaf": 20,
    'min_samples_split': 2
    
}

#Perform Grid Search
grid_search_3 = GridSearchCV(
    estimator=DecisionTreeRegressor(**base_params),
    param_grid=param_grid_3,
    cv=3,  
    verbose=2,  
    n_jobs=-1  
)

#Fit the grid search with training data
grid_search_3.fit(X_train, y_train)

#Get the best parameters and combine them with the base parameters
best_params_final = {**base_params, **grid_search_3.best_params_}

#Output best parameters
print(f"Best parameters after tuning min_samples_split and min_samples_leaf: {best_params_final}")


# In[41]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#Test more values of max_depth based on previous best parameter 
param_grid_4 = {
    'max_depth': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Values to test for min_samples_split
}

base_params = {
    'max_depth': 30,
    'random_state': 42,
    "min_samples_leaf": 20,
    'min_samples_split': 2
    
}

#Perform Grid Search
grid_search_4 = GridSearchCV(
    estimator=DecisionTreeRegressor(**base_params),
    param_grid=param_grid_4,
    cv=3,  # Cross-validation folds
    verbose=2,  # Display progress during search
    n_jobs=-1  # Use all available processors
)

#Fit the grid search with training data
grid_search_4.fit(X_train, y_train)

#Get the best parameters and combine them with the base parameters
best_params_final = {**base_params, **grid_search_4.best_params_}

# Output best parameters
print(f"Best parameters after tuning min_samples_split and min_samples_leaf: {best_params_final}")


# **Metrics**  
# 
# Testing the performance of the model
# 
# Finding Most Important Features
# 
# plotting Predicted Vs Actual Prices

# In[42]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Initialise the Decision Tree Regressor with the best parameters
best_params = {
    'min_samples_leaf': 20,
    'min_samples_split': 2,
    'max_depth': 11
}

dt_model = DecisionTreeRegressor(
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    max_depth=best_params['max_depth'],
    random_state=42
)

# Step 2: Fit the model to your training data
dt_model.fit(X_train, y_train)

# Step 3: Predict on the test data
dt_y_pred = dt_model.predict(X_test)

# Step 4: Calculate MAE, MSE, and R² for Decision Tree
dt_mae = mean_absolute_error(y_test, dt_y_pred)
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_r2 = r2_score(y_test, dt_y_pred)

# Calculate Adjusted R² for Decision Tree
n = len(y_test)  # Number of data points
p = X_test.shape[1]  # Number of features
dt_r2_adj = 1 - ((1 - dt_r2) * (n - 1)) / (n - p - 1)

# Step 5: Print the results for Decision Tree
print(f"Decision Tree - Mean Absolute Error (MAE): {dt_mae}")
print(f"Decision Tree - Mean Squared Error (MSE): {dt_mse}")
print(f"Decision Tree - R² Score: {dt_r2}")
print(f"Decision Tree - Adjusted R²: {dt_r2_adj}")


# In[43]:


# Feature Importances
dt_feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 10 Feature Importances (Decision Tree):")
print(dt_feature_importances.head(10))


# In[44]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Predict using the Decision Tree model
dt_y_pred = dt_model.predict(X_test)

# Calculate Mean Squared Error (MSE) for confidence interval
dt_mse = mean_squared_error(y_test, dt_y_pred)

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', linestyle='-', marker='o', color='blue')
plt.plot(y_test.index, dt_y_pred, label='Predicted Prices', linestyle='--', marker='x', color='orange')

# Calculate confidence interval based on MSE
confidence_interval_upper = dt_y_pred + (dt_mse ** 0.5)
confidence_interval_lower = dt_y_pred - (dt_mse ** 0.5)

# Plot Confidence Interval
plt.fill_between(y_test.index, confidence_interval_lower, confidence_interval_upper, color='green', alpha=0.2, label='Confidence Interval')

# Labels and title
plt.xlabel('Date')
plt.ylabel('Day-Ahead Price (EUR)')
plt.title('Actual vs Predicted Day-Ahead Prices with Decision Tree')

# Add legend and grid
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


# # **Random Forest**

# **Fine Tuning**

# In[76]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Testing different values for min_samples_split and min_samples_leaf
param_grid_1 = {
    'min_samples_split': [3, 5, 7],
    'min_samples_leaf': [4, 5, 6]
}

#initialise the Tree parameters
best_params = {
    'n_estimators': 200,
    'max_depth': 30,
    'bootstrap': True
}


#Perform Grid Search
grid_search_1 = GridSearchCV(
    estimator=RandomForestRegressor(
        n_estimators=best_params['n_estimators'],  # Using best_params['n_estimators']
        max_depth=best_params['max_depth'],  # Using best_params['max_depth']
        bootstrap=best_params['bootstrap'],  # Using best_params['bootstrap']
        random_state=42
    ),
    param_grid=param_grid_1,
    cv=3,
    verbose=2,
    n_jobs=-1
)

#Fit the grid search with training data
grid_search_1.fit(X_train, y_train)

#print best parameters
best_params_final = {**best_params, **grid_search_1.best_params_}
print(f"Best parameters after second round: {best_params_final}")


# In[78]:


#testing more values for tune min_samples_split and min_samples_leaf based on previous results
param_grid_2 = {
    'min_samples_split': [3, 4],
    'min_samples_leaf': [6, 8, 10]
}

#initialise the Tree parameters
best_params = {
    'n_estimators': 200,
    'max_depth': 30,
    'bootstrap': True
}

#Perform Grid Search
grid_search_2 = GridSearchCV(
    estimator=RandomForestRegressor(
        n_estimators=best_params['n_estimators'],  
        max_depth=best_params['max_depth'], 
        bootstrap=best_params['bootstrap'],
        random_state=42
    ),
    param_grid=param_grid_2,
    cv=3,
    verbose=2,
    n_jobs=-1
)

#Fit the grid search with training data
grid_search_2.fit(X_train, y_train)

#print best parameters
best_params_final = {**best_params, **grid_search_2.best_params_}
print(f"Best parameters after second round: {best_params_final}")


# In[81]:


#using the best parameter found for min_samples_split and testing more min_samples_leaf
param_grid_3 = {
    'min_samples_leaf': [10, 20, 50]
}

#Initialising the base parameters
best_params = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 3, #Keeping this value to avoid overfitting
    'bootstrap': True
}

#Perform Grid Search
grid_search_3 = GridSearchCV(
    estimator=RandomForestRegressor(
        min_samples_split=best_params['min_samples_split'],  # Using best_params['n_estimators']
        n_estimators=best_params['n_estimators'],  # Using best_params['n_estimators']
        max_depth=best_params['max_depth'],  # Using best_params['max_depth']
        bootstrap=best_params['bootstrap'],  # Using best_params['bootstrap']
        random_state=42
    ),
    param_grid=param_grid_3,
    cv=3,
    verbose=2,
    n_jobs=-1
)

#Fit the grid search with training data
grid_search_3.fit(X_train, y_train)

#print best parameters
best_params_final = {**best_params, **grid_search_3.best_params_}
print(f"Best parameters after second round: {best_params_final}")


# In[82]:


param_grid_4 = {
    'bootstrap': [True, False]
}

#Initialising the base parameters
best_params = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 3,
    'min_samples_leaf': 10
}

grid_search_4 = GridSearchCV(
    estimator=RandomForestRegressor(
        min_samples_split=3,  
        n_estimators=200,  
        max_depth=30,  
        min_samples_leaf = 10,
        random_state=42
    ),
    param_grid=param_grid_4,
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Fit the grid search with training data
grid_search_4.fit(X_train, y_train)

# Final best parameters after the second round of tuning
best_params_final = {**best_params, **grid_search_4.best_params_}
print(f"Best parameters after second round: {best_params_final}")


# In[ ]:


#Testing different values for n_estimators and max_depth
param_grid_5 = {
    'n_estimators': [200, 250, 300],  # Different values for n_estimators
    'max_depth': [30, 40, 50]   # Different values for max_depth
}

#Initialising with base parameters based on best parameters found so far
best_params = {
    'min_samples_split': 3,
    'min_samples_leaf': 10,
    'bootstrap': True
}

#Perform Grid Search
grid_search_5 = GridSearchCV(
    estimator=RandomForestRegressor(
        min_samples_split=3,   
        min_samples_leaf = 10,
        bootstrap = True,
        random_state=42
    ),
    param_grid=param_grid_5,
    cv=3,
    verbose=2,
    n_jobs=-1
)

#Fit the grid search with training data
grid_search_5.fit(X_train, y_train)

#Final best parameters after the second round of tuning
best_params_final = {**best_params, **grid_search_5.best_params_}
print(f"Best parameters after second round: {best_params_final}")


# In[ ]:


param_grid_6 = {
    'n_estimators': [300, 250, 400]
}

best_params = {
    'max_depth': 30,
    'min_samples_split': 3,
    'min_samples_leaf': 10,
    'bootstrap': True
}


grid_search_6 = GridSearchCV(
    estimator=RandomForestRegressor(
        min_samples_split=3,   
        min_samples_leaf = 10,
        bootstrap = True,
        max_depth = 30,
        random_state=42
    ),
    param_grid=param_grid_6,
    cv=3,
    verbose=2,
    n_jobs=-1
)

#Fit the grid search with training data
grid_search_6.fit(X_train, y_train)

#Final best parameters after the second round of tuning
best_params_final = {**best_params, **grid_search_6.best_params_}
print(f"Best parameters after second round: {best_params_final}")


# In[ ]:


param_grid_7 = {
    'n_estimators': [500, 550, 600]
}

best_params = {
    'max_depth': 30,
    'min_samples_split': 3,
    'min_samples_leaf': 10,
    'bootstrap': True
}

grid_search_7 = GridSearchCV(
    estimator=RandomForestRegressor(
        min_samples_split=3,   
        min_samples_leaf = 10,
        bootstrap = True,
        max_depth = 30,
        random_state=42
    ),
    param_grid=param_grid_7,
    cv=3,
    verbose=2,
    n_jobs=-1
)

#Fit the grid search with training data
grid_search_7.fit(X_train, y_train)

#Final best parameters after the second round of tuning
best_params_final = {**best_params, **grid_search_7.best_params_}
print(f"Best parameters after second round: {best_params_final}")


# **Metrics**  
# 
# Testing the performance of the model
# 
# Finding Most Important Features
# 
# plotting Predicted Vs Actual Prices

# In[83]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Define the best parameters found from GridSearchCV or RandomizedSearchCV
rf_best_params = {
    'n_estimators': 500,
    'max_depth': 30,
    'min_samples_split': 3,
    'min_samples_leaf': 10,
    'bootstrap': True
}

#Initialise the RandomForestRegressor with the best parameters
rf_model = RandomForestRegressor(
    n_estimators=rf_best_params['n_estimators'],
    max_depth=rf_best_params['max_depth'],
    min_samples_split=rf_best_params['min_samples_split'],
    min_samples_leaf=rf_best_params['min_samples_leaf'],
    bootstrap=rf_best_params['bootstrap'],
    random_state=42  # Optional: To ensure reproducibility
)

#Fit the model on the training data
rf_model.fit(X_train, y_train)

#Predict on the test data (optional, for evaluation)
rf_y_pred = rf_model.predict(X_test)

#Evaluate the model's performance (optional)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

#Calculate Adjusted R² for Random Forest
n = len(y_test)  # Number of data points
p = X_test.shape[1]  # Number of features
rf_r2_adj = 1 - ((1 - rf_r2) * (n - 1)) / (n - p - 1)

#Print the results for Random Forest
print(f"Random Forest - Mean Squared Error (MSE): {rf_mse}")
print(f"Random Forest - Mean Absolute Error (MAE): {rf_mae}")
print(f"Random Forest - R² Score: {rf_r2}")
print(f"Random Forest - Adjusted R²: {rf_r2_adj}")


# In[85]:


#Feature Importances for Random Forest
rf_feature_importances = pd.DataFrame({
    'Feature': X_train.columns,  # Ensure X_train is a DataFrame with column names
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

#Display the top 10 features
print("Top 10 Feature Importances:")
print(rf_feature_importances.head(10))


# In[84]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Predict using the Random Forest model
rf_y_pred = rf_model.predict(X_test)

#Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', linestyle='-', marker='o', color='blue')
plt.plot(y_test.index, rf_y_pred, label='Predicted Prices', linestyle='--', marker='x', color='orange')

#Calculate confidence interval based on MSE
confidence_interval_upper = rf_y_pred + (rf_mse ** 0.5)
confidence_interval_lower = rf_y_pred - (rf_mse ** 0.5)

#Plot Confidence Interval
plt.fill_between(y_test.index, confidence_interval_lower, confidence_interval_upper, color='green', alpha=0.2, label='Confidence Interval')

# Labels and title
plt.xlabel('Date')
plt.ylabel('Day-Ahead Price (EUR)')
plt.title('Actual vs Predicted Day-Ahead Prices with Random Forest')

#Add legend and grid
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

#Show the plot
plt.show()


# # **XGBoost**

# **Fine Tuning**

# In[86]:


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

#Initialise the model with fixed parameters
fixed_params = {
    'learning_rate': 0.1,  # Fixed
    'n_estimators': 1000,  # Fixed
    'max_depth': 5,  # Fixed
    'min_child_weight': 1,  # Fixed
    'gamma': 0,  # Fixed
    'subsample': 0.8,  # Fixed
    'colsample_bytree': 0.8,  # Fixed
    'nthread': 4,  # Fixed
    'scale_pos_weight': 1,  # Fixed
    'seed': 27  # Fixed
}

xgb_model = xgb.XGBRegressor(**fixed_params)


# In[87]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid for tuning max_depth and min_child_weight
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,  # Cross-validation
    verbose=2,  # Detailed progress messages
    n_jobs=-1  # Use all available processors
)

# Fit the model on the training data
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")


# In[88]:


# Define the parameter grid for tuning max_depth and min_child_weight
param_grid = {
    'max_depth': [2, 3, 4],
    'min_child_weight': [1, 2]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,  # Cross-validation
    verbose=2,  # Detailed progress messages
    n_jobs=-1  # Use all available processors
)

# Fit the model on the training data
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")


# In[89]:


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

#Initialize the model with fixed parameters
fixed_params = {
    'learning_rate': 0.1,  # Fixed
    'n_estimators': 1000,  # Fixed
    'max_depth': 3,  # Fixed
    'min_child_weight': 2,  # Fixed
    'gamma': 0,  # Fixed
    'subsample': 0.8,  # Fixed
    'colsample_bytree': 0.8,  # Fixed
    'nthread': 4,  # Fixed
    'scale_pos_weight': 1,  # Fixed
    'seed': 27  # Fixed
}

xgb_model = xgb.XGBRegressor(**fixed_params)


# In[90]:


from sklearn.model_selection import GridSearchCV

#Define the parameter grid for gamma
param_test = {
    'gamma': [i/10.0 for i in range(0, 5)]  # Testing values: [0.0, 0.1, 0.2, 0.3, 0.4]
}

#Perform GridSearchCV with XGBRegressor and your defined parameter grid
grid_search = GridSearchCV(
    estimator=xgb_model,  # xgb_model should already be initialized with your fixed parameters
    param_grid=param_test,
    cv=3,  # Cross-validation
    verbose=2,
    n_jobs=-1
)

#Fit the model to your training data
grid_search.fit(X_train, y_train)

#Get the best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best parameters for gamma: {grid_search.best_params_}")


# In[91]:


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

#Initialise the model with fixed parameters
fixed_params = {
    'learning_rate': 0.1,  # Fixed
    'n_estimators': 1000,  # Fixed
    'max_depth': 3,  # Fixed
    'min_child_weight': 2,  # Fixed
    'gamma': 0.3,  # Fixed
    'subsample': 0.8,  # Fixed
    'colsample_bytree': 0.8,  # Fixed
    'nthread': 4,  # Fixed
    'scale_pos_weight': 1,  # Fixed
    'seed': 27  # Fixed
}

xgb_model = xgb.XGBRegressor(**fixed_params)


# In[92]:


param_test = {
    'subsample': [i/10.0 for i in range(6, 10)],  # [0.6, 0.7, 0.8, 0.9]
    'colsample_bytree': [i/10.0 for i in range(6, 10)]  # [0.6, 0.7, 0.8, 0.9]
}

#Perform GridSearchCV with XGBRegressor and your defined parameter grid
grid_search = GridSearchCV(
    estimator=xgb_model,  # xgb_model should already be initialized with your fixed parameters
    param_grid=param_test,
    cv=3,  # Cross-validation
    verbose=2,
    n_jobs=-1
)

#Fit the model to your training data
grid_search.fit(X_train, y_train)

#Get the best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best parameters for subsample and colsample_bytree: {grid_search.best_params_}")


# In[93]:


param_test = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

#Perform GridSearchCV with XGBRegressor and your defined parameter grid
grid_search = GridSearchCV(
    estimator=xgb_model,  # xgb_model should already be initialized with your fixed parameters
    param_grid=param_test,
    cv=3,  # Cross-validation
    verbose=2,
    n_jobs=-1
)

#Fit the model to your training data
grid_search.fit(X_train, y_train)

#Get the best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best parameters for subsample and colsample_bytree: {grid_search.best_params_}")


# In[94]:


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

#Initialise the model with fixed parameters
fixed_params = {
    'learning_rate': 0.1,  # Fixed
    'n_estimators': 1000,  # Fixed
    'max_depth': 3,  # Fixed
    'min_child_weight': 2,  # Fixed
    'gamma': 0.3,  # Fixed
    'subsample': 0.8,  # Fixed
    'colsample_bytree': 0.8,  # Fixed
    'nthread': 4,  # Fixed
    'scale_pos_weight': 1,  # Fixed
    'seed': 27  # Fixed
}

xgb_model = xgb.XGBRegressor(**fixed_params)


# In[95]:


param_test = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}

grid_search = GridSearchCV(
    estimator=xgb_model,  # xgb_model should already be initialized with your fixed parameters
    param_grid=param_test,
    cv=3,  # Cross-validation
    verbose=2,
    n_jobs=-1
)

#Fit the model to your training data
grid_search.fit(X_train, y_train)

#Get the best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best parameters for subsample and colsample_bytree: {grid_search.best_params_}")


# In[96]:


param_test = {
    'reg_alpha': [10, 50, 75, 100, 125, 150, 175]
}

grid_search = GridSearchCV(
    estimator=xgb_model,  # xgb_model should already be initialized with your fixed parameters
    param_grid=param_test,
    cv=3,  # Cross-validation
    verbose=2,
    n_jobs=-1
)

#Fit the model to your training data
grid_search.fit(X_train, y_train)

#Get the best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best parameters for subsample and colsample_bytree: {grid_search.best_params_}")


# In[97]:


param_test = {
    'reg_alpha': [115, 120, 125]
}

grid_search = GridSearchCV(
    estimator=xgb_model,  # xgb_model should already be initialized with your fixed parameters
    param_grid=param_test,
    cv=3,  # Cross-validation
    verbose=2,
    n_jobs=-1
)

#Fit the model to your training data
grid_search.fit(X_train, y_train)

#Get the best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best parameters for subsample and colsample_bytree: {grid_search.best_params_}")


# **Metrics**  
# 
# Testing the performance of the model
# 
# Finding Most Important Features
# 
# plotting Predicted Vs Actual Prices

# In[98]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Define the best parameters found from your GridSearchCV
xgb_best_params = {
    'reg_alpha': 120,  # Best reg_alpha from grid search
    'max_depth': 3,  # Best max_depth from grid search
    'min_child_weight': 2,  # Best min_child_weight from grid search
    'gamma': 0.3,  # Best gamma from grid search
    'colsample_bytree': 0.8,  # Best colsample_bytree from grid search
    'subsample': 0.8  # Best subsample from grid search
}

#Initialise XGBoost model with the best parameters
xgb_model = xgb.XGBRegressor(
    n_estimators=5000,  # You can adjust this depending on your model tuning
    random_state=42,
    learning_rate=0.01,  # Best parameter for learning_rate
    reg_alpha=xgb_best_params['reg_alpha'],  # Add the best reg_alpha
    max_depth=xgb_best_params['max_depth'],  # Add the best max_depth
    min_child_weight=xgb_best_params['min_child_weight'],  # Add the best min_child_weight
    gamma=xgb_best_params['gamma'],  # Add the best gamma
    colsample_bytree=xgb_best_params['colsample_bytree'],  # Add the best colsample_bytree
    subsample=xgb_best_params['subsample']  # Add the best subsample
)

#Train the model
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)

#Calculate performance metrics
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_mae = mean_absolute_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)

#Calculate Adjusted R²
n = len(y_test)  # Number of data points
p = X_test.shape[1]  # Number of features
xgb_r2_adj = 1 - ((1 - xgb_r2) * (n - 1)) / (n - p - 1)

#Print the results
print(f"XGBoost - Mean Squared Error (MSE): {xgb_mse}")
print(f"XGBoost - Mean Absolute Error (MAE): {xgb_mae}")
print(f"XGBoost - R² Score: {xgb_r2}")
print(f"XGBoost - Adjusted R²: {xgb_r2_adj}")


# In[99]:


# Feature Importances
xgb_feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 10 Feature Importances:")
print(xgb_feature_importances.head(10))


# In[100]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', linestyle='-', marker='o', color='blue')
plt.plot(y_test.index, xgb_y_pred, label='Predicted Prices', linestyle='--', marker='x', color='orange')

# Calculate confidence interval based on MSE
confidence_interval_upper = xgb_y_pred + (xgb_mse ** 0.5)
confidence_interval_lower = xgb_y_pred - (xgb_mse ** 0.5)

# Plot Confidence Interval
plt.fill_between(y_test.index, confidence_interval_lower, confidence_interval_upper, color='green', alpha=0.2, label='Confidence Interval')

# Labels and title
plt.xlabel('Date')
plt.ylabel('Day-Ahead Price (EUR)')
plt.title('Actual vs Predicted Day-Ahead Prices with XGBoost')

# Add legend and grid
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


# # **SVM**

# In[101]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Ensure your training and testing data are already defined (X_train, X_test, y_train, y_test)

# 1.Scaling the Data: Standardize features for better performance with SVM.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2.Train the Support Vector Machine (SVM) model
svm_model = SVR(kernel='rbf')  # You can change the kernel if needed (e.g., 'linear', 'poly', 'rbf')
svm_model.fit(X_train_scaled, y_train)

# 3.Make predictions using the trained model
y_pred_svm = svm_model.predict(X_test_scaled)

# 4.Calculate Mean Squared Error (MSE)
mse_svm = mean_squared_error(y_test, y_pred_svm)

# 5.Calculate Mean Absolute Error (MAE)
mae_svm = mean_absolute_error(y_test, y_pred_svm)

# 6.Calculate R-squared (R²)
r2_svm = r2_score(y_test, y_pred_svm)

# 7.Calculate Adjusted R-squared
n = len(y_test)  # Number of test samples
p = X_test.shape[1]  # Number of features
adj_r2_svm = 1 - (1 - r2_svm) * (n - 1) / (n - p - 1)

#Print results
print(f"SVM - Mean Squared Error (MSE): {mse_svm}")
print(f"SVM - Mean Absolute Error (MAE): {mae_svm}")
print(f"SVM - R-squared (R²): {r2_svm}")
print(f"SVM - Adjusted R-squared (Adjusted R²): {adj_r2_svm}")


print("performing cross validation...")
# 8.Cross-validation to evaluate the model performance
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-cv_scores.mean()} (+/- {cv_scores.std()})")

# 9.Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.2, 0.3]
}
print("performing grid search...")
grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
print("Best parameters found: ", grid_search.best_params_)

# 10.Plot Actual vs Predicted Prices (for visualization)
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', linestyle='-', marker='o', color='blue')
plt.plot(y_test.index, y_pred_svm, label='Predicted Prices', linestyle='--', marker='x', color='orange')
plt.fill_between(y_test.index, y_pred_svm - mse_svm**0.5, y_pred_svm + mse_svm**0.5, color='green', alpha=0.2, label='Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Day-Ahead Price (EUR)')
plt.title('Actual vs Predicted Day-Ahead Prices with SVM')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[108]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#Ensure your training and testing data are already defined (X_train, X_test, y_train, y_test)

# 1.Scaling the Data**: Standardize features for better performance with SVM.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2.Train the Support Vector Machine (SVM) model
svm_model = SVR(kernel='rbf', C=100, epsilon=0.01, gamma=0.001)  # You can change the kernel if needed (e.g., 'linear', 'poly', 'rbf')
svm_model.fit(X_train_scaled, y_train)

# 3.Make predictions using the trained model
y_pred_svm = svm_model.predict(X_test_scaled)

# 4.Calculate Mean Squared Error (MSE)
mse_svm = mean_squared_error(y_test, y_pred_svm)

# 5.Calculate Mean Absolute Error (MAE)
mae_svm = mean_absolute_error(y_test, y_pred_svm)

# 6.Calculate R-squared (R²)
r2_svm = r2_score(y_test, y_pred_svm)

# 7.Calculate Adjusted R-squared
n = len(y_test)  # Number of test samples
p = X_test.shape[1]  # Number of features
adj_r2_svm = 1 - (1 - r2_svm) * (n - 1) / (n - p - 1)

# Print results
print(f"SVM - Mean Squared Error (MSE): {mse_svm}")
print(f"SVM - Mean Absolute Error (MAE): {mae_svm}")
print(f"SVM - R-squared (R²): {r2_svm}")
print(f"SVM - Adjusted R-squared (Adjusted R²): {adj_r2_svm}")


# In[102]:


from sklearn.inspection import permutation_importance

#Calculate permutation importance
result = permutation_importance(svm_model, X_test_scaled, y_test, n_repeats=10, random_state=42)

#Organise results into a DataFrame
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

#Display top 10 features
print("Top Feature Importances:")
print(feature_importances.head(10))


# In[109]:


y_pred_svm = svm_model.predict(X_test_scaled)

mse_svm = mean_squared_error(y_test, y_pred_svm)
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', linestyle='-', marker='o', color='blue')
plt.plot(y_test.index, y_pred_svm, label='Predicted Prices', linestyle='--', marker='x', color='orange')

confidence_interval_upper = y_pred_svm + (mse_svm ** 0.5)
confidence_interval_lower = y_pred_svm - (mse_svm ** 0.5)

plt.fill_between(y_test.index, confidence_interval_lower, confidence_interval_upper, color='green', alpha=0.2, label='Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Day-Ahead Price (EUR)')
plt.title('Actual vs Predicted Day-Ahead Prices with SVM')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




