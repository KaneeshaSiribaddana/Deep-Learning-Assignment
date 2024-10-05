import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # Library to save and load models

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    print(data.head())
    return data

def preprocess_data(data):
    """Preprocess the data."""
    # Fill missing values in numeric columns with the mean
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)

    return data
    
def split_data(data, target_variable):
    """Split the dataset into features and target variable, then into training and test sets."""
    X = data.drop([target_variable], axis=1)
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def plot_feature_importance(model, feature_names):
    """Plot feature importance of the model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create a bar chart
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    
    # Limit to the top N features (for example, top 10)
    top_n = 10
    plt.bar(range(top_n), importances[indices][:top_n], align="center")
    
    # Adjust x-ticks
    plt.xticks(range(top_n), feature_names[indices][:top_n], rotation=45, ha='right')
    
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.show()

def train_model(X_train, y_train):
    """Train the Random Forest Regressor model."""
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and return performance metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

def main():
    # Load the dataset
    file_path = 'AmesHousing.csv' 
    data = load_data(file_path)

    # Preprocess the data
    data = preprocess_data(data)

    # Split the data
    target_variable = 'SalePrice'  # Specify the target variable
    X_train, X_test, y_train, y_test = split_data(data, target_variable)

    # Train the model
    rf_model = train_model(X_train, y_train)

    # Evaluate the model
    mae, mse, r2 = evaluate_model(rf_model, X_test, y_test)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Plot feature importance
    plot_feature_importance(rf_model, X_train.columns)

    # Save the model
    joblib.dump(rf_model, 'random_forest_model.joblib')

if __name__ == "__main__":
    main()
