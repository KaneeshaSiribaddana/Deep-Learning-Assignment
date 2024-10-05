import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # Library to save and load models
import seaborn as sns
import time

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

def tune_hyperparameters(X_train, y_train):
    """Tune hyperparameters using GridSearchCV."""
    rf_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }

    start_time = time.time()  # Start time
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    end_time = time.time()  # End time
    
    print(f"Hyperparameter tuning time: {end_time - start_time:.2f} seconds")
    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_

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

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and return performance metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

def plot_residuals(y_test, y_pred):
    """Plot the residuals to evaluate model performance visually."""
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()

def main():
    # Load the dataset
    file_path = 'AmesHousing.csv' 
    data = load_data(file_path)

    # Preprocess the data
    data = preprocess_data(data)

    # Split the data
    target_variable = 'SalePrice'  # Specify the target variable
    X_train, X_test, y_train, y_test = split_data(data, target_variable)

    # Hyperparameter tuning
    rf_model = tune_hyperparameters(X_train, y_train)

    # Evaluate the model
    mae, mse, r2 = evaluate_model(rf_model, X_test, y_test)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Plot feature importance
    plot_feature_importance(rf_model, X_train.columns)

    # Plot residuals
    y_pred = rf_model.predict(X_test)
    plot_residuals(y_test, y_pred)

    # Save the model
    joblib.dump(rf_model, 'random_forest_model.joblib')

if __name__ == "__main__":
    main()
