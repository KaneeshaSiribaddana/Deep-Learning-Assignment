import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Load dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(columns=['Order', 'PID'])  # Drop unnecessary columns
    print(data.columns)
    return data

# Feature Engineering
def feature_engineering(data):
    data['TotalRooms'] = data['Bedroom AbvGr'] + data['Full Bath'] + data['Half Bath']
    # Additional feature engineering can go here
    return data

# Data Preprocessing
def preprocess_data(data):
    data = feature_engineering(data)  # Perform feature engineering
    X = data.drop(columns=['SalePrice'])  # Features
    y = data['SalePrice']  # Target

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Add scaler to standardize numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                     ('scaler', StandardScaler())]), numerical_features),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                     ('onehot', OneHotEncoder(handle_unknown='ignore'))]), 
             categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test

# Build the Deep Neural Network model
def build_model(input_shape):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_shape,)))  # Increased neurons
    model.add(Dropout(0.3))  # Increased dropout rate
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=200, batch_size=32):  # Increased epochs
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size)
    return history

# Visualize learning curves
def plot_learning_curves(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R^2 Score: {r2}")

    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
    print(results.head(10))  # Display the first 10 predictions

    plot_actual_vs_predicted(y_test, y_pred)

    return mae, mse, r2

# Plot Actual vs Predicted values
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Reference line
    plt.title('Actual vs Predicted Sale Prices')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.axis('equal')
    plt.grid()
    plt.show()

# Main function to add everything
def main():
    filepath = 'AmesHousing.csv' 
    data = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = build_model(X_train.shape[1])
    history = train_model(model, X_train, y_train, epochs=200, batch_size=32)
    plot_learning_curves(history)
    evaluate_model(model, X_test, y_test)
    model.save('saved_model/DNN_model') 

# Run the main function
if __name__ == "__main__":
    main()
