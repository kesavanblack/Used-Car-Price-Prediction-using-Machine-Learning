# Used Car Price Prediction using Machine Learning

## 🎯 Project Overview

This project implements a comprehensive machine learning solution for predicting used car prices. The system helps both buyers and sellers make informed decisions by providing accurate price estimates based on various car attributes.

## 📊 Dataset Features

### Input Features
- **Brand**: Car manufacturer (Maruti, Honda, Toyota, etc.)
- **Model**: Specific car model (Swift, City, Innova, etc.)
- **Year**: Manufacturing year (2010-2023)
- **Kilometers Driven**: Total distance covered
- **Fuel Type**: Petrol, Diesel, CNG, Electric
- **Transmission**: Manual or Automatic
- **Ownership**: First, Second, Third, Fourth & Above

### Target Variable
- **Selling Price**: Market price of the used car (in INR)

## 🔧 Technical Implementation

### 1. Data Preprocessing Pipeline

#### Missing Value Handling
- Identification and removal of null values
- Data quality assessment and cleaning

#### Feature Engineering
```python
# Age calculation
df['age'] = 2024 - df['year']

# Ownership encoding (ordinal)
ownership_mapping = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}

# One-hot encoding for categorical features
pd.get_dummies(df, columns=['brand', 'model', 'fuel_type', 'transmission'])
```

#### Feature Scaling
- StandardScaler for linear models (Linear Regression, SVR)
- Raw features for tree-based models (Random Forest, XGBoost)

### 2. Machine Learning Models

#### Model Selection Strategy
We implemented and compared four different regression algorithms:

1. **Linear Regression**
   - Baseline model for linear relationships
   - Interpretable coefficients
   - Fast training and prediction

2. **Random Forest Regressor**
   - Ensemble method with multiple decision trees
   - Handles non-linear relationships
   - Built-in feature importance
   - Robust to overfitting

3. **XGBoost Regressor**
   - Gradient boosting algorithm
   - Superior performance on structured data
   - Advanced regularization techniques
   - Excellent generalization

4. **Support Vector Regression (SVR)**
   - Kernel-based approach (RBF kernel)
   - Effective for high-dimensional data
   - Memory efficient

#### Hyperparameter Tuning
```python
# Random Forest Parameters
RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

# XGBoost Parameters
XGBRegressor(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)
```

### 3. Model Evaluation Metrics

#### Primary Metrics
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual prices
- **Mean Squared Error (MSE)**: Average squared difference (penalizes larger errors)
- **Root Mean Squared Error (RMSE)**: Square root of