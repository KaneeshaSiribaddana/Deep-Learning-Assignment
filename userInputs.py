import pandas as pd
import joblib
import numpy as np
import tkinter as tk
from tkinter import messagebox

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

def predict_price(user_input):
    """Make prediction based on user input."""
    # Load the model
    model_path = 'random_forest_model.joblib'
    rf_model = load_model(model_path)

    # Get the full feature names from the model
    all_features = rf_model.feature_names_in_

    # Create a dictionary for all features, using user input and default values for missing features
    user_input_full = {feature: 0.0 for feature in all_features}  # Default all to 0
    user_input_full.update(user_input)  # Update with user input

    # Convert user input to DataFrame for prediction
    user_input_df = pd.DataFrame([user_input_full])

    # Make prediction
    predicted_price = rf_model.predict(user_input_df)
    return predicted_price[0]

def submit():
    """Handle the submit button action."""
    user_input = {}
    for feature in top_features:
        try:
            user_input[feature] = float(entries[feature].get())
        except ValueError:
            messagebox.showerror("Input Error", f"Please enter a valid number for {feature}.")
            return
    
    predicted_price = predict_price(user_input)
    messagebox.showinfo("Predicted Price", f"The predicted sale price of the house is: ${predicted_price:,.2f}")

# Load the model and get the top features
model_path = 'random_forest_model.joblib'
rf_model = load_model(model_path)
top_features, top_indices = get_top_n_features(rf_model, n=10)

# Create the main window
root = tk.Tk()
root.title("House Price Prediction")
root.geometry("400x400")
root.configure(bg="#f0f0f0")  # Light background color

# Create a heading label
heading = tk.Label(root, text="House Price Prediction", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
heading.grid(row=0, columnspan=2, pady=10)  # Changed to grid()

# Create a dictionary to hold entry widgets
entries = {}

# Create labels and entry fields for each feature with better naming
feature_labels = {
    "feature1": "Square Footage (in sqft):",
    "feature2": "Number of Bedrooms:",
    "feature3": "Number of Bathrooms:",
    "feature4": "Lot Size (in acres):",
    "feature5": "Year Built:",
    "feature6": "Garage Size (in cars):",
    "feature7": "Location Rating (1-10):",
    "feature8": "Walkability Score (1-10):",
    "feature9": "Nearby School Rating (1-10):",
    "feature10": "HOA Fees (monthly):",
}

# Dynamically generate the input fields based on the top features
for idx, feature in enumerate(top_features):
    label_text = feature_labels.get(feature, f"Enter value for {feature}:")
    label = tk.Label(root, text=label_text, font=("Helvetica", 12), bg="#f0f0f0")
    label.grid(row=idx + 1, column=0, padx=10, pady=5, sticky='e')  # Align labels to the right
    
    entry = tk.Entry(root, font=("Helvetica", 12))
    entry.grid(row=idx + 1, column=1, padx=10, pady=5)
    entries[feature] = entry

# Create the submit button
submit_button = tk.Button(root, text="Submit", command=submit, font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white")
submit_button.grid(row=len(top_features) + 1, columnspan=2, pady=20)

# Start the GUI main loop
root.mainloop()
