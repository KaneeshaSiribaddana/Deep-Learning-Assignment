import pandas as pd
import joblib
import numpy as np

def load_model(model_path):
    """Load the trained model from a file."""
    model = joblib.load(model_path)
    return model

def get_top_n_features(model, n=10):
    """Get the top N features based on their importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort indices by importance
    top_indices = indices[:n]  # Get the indices of the top N features
    return model.feature_names_in_[top_indices], top_indices

def get_user_input(feature_names):
    """Get user input from the terminal."""
    user_input = {}
    for feature in feature_names:
        user_input[feature] = float(input(f"Enter value for {feature}: "))
    return user_input

def main():
    # Load the model
    model_path = 'random_forest_model.joblib'
    rf_model = load_model(model_path)

    # Get the top 10 features from the model
    top_features, top_indices = get_top_n_features(rf_model, n=10)

    # Get user input only for the top features
    user_input = get_user_input(top_features)

    # Convert user input to DataFrame for prediction
    user_input_df = pd.DataFrame([user_input])

    # Make prediction
    predicted_price = rf_model.predict(user_input_df)
    print(f"The predicted sale price of the house is: ${predicted_price[0]:,.2f}")

if __name__ == "__main__":
    main()
