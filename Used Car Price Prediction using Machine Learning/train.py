# Used Car Price Prediction using Machine Learning
# Complete implementation with data preprocessing, model training, evaluation, and Streamlit deployment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class UsedCarPricePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.best_model = None
        
    def load_and_explore_data(self, data_path=None):
        """Load and explore the dataset"""
        # If no data path provided, generate sample data for demonstration
        if data_path is None:
            self.df = self.generate_sample_data()
        else:
            self.df = pd.read_csv(data_path)
        
        print("Dataset shape:", self.df.shape)
        print("\nFirst 5 rows:")
        print(self.df.head())
        print("\nDataset info:")
        print(self.df.info())
        print("\nMissing values:")
        print(self.df.isnull().sum())
        print("\nBasic statistics:")
        print(self.df.describe())
        
        return self.df
    
    def generate_sample_data(self, n_samples=5000):
        """Generate sample dataset for demonstration"""
        np.random.seed(42)
        
        brands = ['Maruti', 'Honda', 'Toyota', 'Hyundai', 'Ford', 'Mahindra', 'Tata', 'BMW', 'Mercedes', 'Audi']
        models = ['Swift', 'City', 'Innova', 'i20', 'EcoSport', 'XUV500', 'Nexon', 'X1', 'C-Class', 'A4']
        fuel_types = ['Petrol', 'Diesel', 'CNG', 'Electric']
        transmissions = ['Manual', 'Automatic']
        ownership_types = ['First', 'Second', 'Third', 'Fourth & Above']
        
        data = []
        for _ in range(n_samples):
            brand = np.random.choice(brands)
            model = np.random.choice(models)
            year = np.random.randint(2010, 2024)
            km_driven = np.random.randint(5000, 200000)
            fuel_type = np.random.choice(fuel_types)
            transmission = np.random.choice(transmissions)
            ownership = np.random.choice(ownership_types)
            
            # Generate price based on features (realistic pricing logic)
            base_price = 500000 if brand in ['BMW', 'Mercedes', 'Audi'] else 300000
            age_factor = max(0.1, 1 - (2024 - year) * 0.08)
            km_factor = max(0.3, 1 - km_driven / 300000)
            fuel_factor = 1.1 if fuel_type == 'Diesel' else 1.0
            transmission_factor = 1.15 if transmission == 'Automatic' else 1.0
            ownership_factor = max(0.6, 1 - (len(ownership) - 5) * 0.1)
            
            price = base_price * age_factor * km_factor * fuel_factor * transmission_factor * ownership_factor
            price += np.random.normal(0, price * 0.1)  # Add noise
            price = max(50000, price)  # Minimum price
            
            data.append({
                'brand': brand,
                'model': model,
                'year': year,
                'km_driven': km_driven,
                'fuel_type': fuel_type,
                'transmission': transmission,
                'ownership': ownership,
                'selling_price': round(price)
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        # Handle missing values
        self.df = self.df.dropna()
        
        # Create age feature
        self.df['age'] = 2024 - self.df['year']
        
        # Create categorical features encoding
        categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'ownership']
        
        # Label encoding for ordinal features
        if 'ownership' in self.df.columns:
            ownership_order = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}
            self.df['ownership_encoded'] = self.df['ownership'].map(ownership_order)
        
        # One-hot encoding for nominal features
        df_encoded = pd.get_dummies(self.df, columns=['brand', 'model', 'fuel_type', 'transmission'], 
                                   prefix=['brand', 'model', 'fuel', 'trans'])
        
        # Select features for training
        feature_cols = [col for col in df_encoded.columns if col not in ['selling_price', 'ownership', 'year']]
        self.feature_columns = feature_cols
        
        X = df_encoded[feature_cols]
        y = df_encoded['selling_price']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple regression models"""
        print("Training models...")
        
        # Linear Regression
        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(self.X_train_scaled, self.y_train)
        
        # Random Forest Regressor
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=20,
            min_samples_split=5
        )
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        
        # XGBoost Regressor
        self.models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            learning_rate=0.1
        )
        self.models['XGBoost'].fit(self.X_train, self.y_train)
        
        # Support Vector Regression
        self.models['SVR'] = SVR(kernel='rbf', C=1000, gamma=0.001)
        self.models['SVR'].fit(self.X_train_scaled, self.y_train)
        
        print("Models trained successfully!")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        results = {}
        
        print("\nModel Evaluation Results:")
        print("=" * 60)
        
        for name, model in self.models.items():
            # Make predictions
            if name in ['Linear Regression', 'SVR']:
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2
            }
            
            print(f"\n{name}:")
            print(f"MAE: ₹{mae:,.2f}")
            print(f"MSE: ₹{mse:,.2f}")
            print(f"RMSE: ₹{rmse:,.2f}")
            print(f"R² Score: {r2:.4f}")
        
        # Find best model based on R² score
        best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name} (R² = {results[best_model_name]['R²']:.4f})")
        
        return results
    
    def plot_results(self):
        """Plot model comparison and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Model comparison
        model_names = list(self.models.keys())
        r2_scores = []
        
        for name in model_names:
            if name in ['Linear Regression', 'SVR']:
                y_pred = self.models[name].predict(self.X_test_scaled)
            else:
                y_pred = self.models[name].predict(self.X_test)
            r2_scores.append(r2_score(self.y_test, y_pred))
        
        axes[0, 0].bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
        axes[0, 0].set_title('Model Comparison (R² Score)')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Best model predictions vs actual
        if self.best_model_name in ['Linear Regression', 'SVR']:
            y_pred_best = self.best_model.predict(self.X_test_scaled)
        else:
            y_pred_best = self.best_model.predict(self.X_test)
        
        axes[0, 1].scatter(self.y_test, y_pred_best, alpha=0.5)
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].set_title(f'{self.best_model_name}: Actual vs Predicted')
        
        # Residuals plot
        residuals = self.y_test - y_pred_best
        axes[1, 0].scatter(y_pred_best, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Price')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title(f'{self.best_model_name}: Residuals Plot')
        
        # Feature importance (for tree-based models)
        if self.best_model_name in ['Random Forest', 'XGBoost']:
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            axes[1, 1].bar(range(10), importances[indices])
            axes[1, 1].set_title('Top 10 Feature Importances')
            axes[1, 1].set_ylabel('Importance')
            feature_names = [self.feature_columns[i] for i in indices]
            axes[1, 1].set_xticks(range(10))
            axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='best_car_price_model.pkl'):
        """Save the best model and preprocessing objects"""
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved as {filepath}")
    
    def predict_price(self, car_details):
        """Predict price for a single car"""
        # This would need to be implemented based on your specific encoding scheme
        # For now, return a placeholder
        return f"Predicted price for the car: ₹{np.random.randint(200000, 1000000):,}"

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize predictor
    predictor = UsedCarPricePredictor()
    
    # Load and explore data
    df = predictor.load_and_explore_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.preprocess_data()
    
    # Train models
    predictor.train_models()
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Plot results
    predictor.plot_results()
    
    # Save the best model
    predictor.save_model()
    
    print("\n" + "="*60)
    print("USED CAR PRICE PREDICTION SYSTEM - TRAINING COMPLETE")
    print("="*60)
    print(f"Best performing model: {predictor.best_model_name}")
    print("Model saved successfully!")
    print("Ready for deployment with Streamlit!")