import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preprocessing import load_and_preprocess_data
from model import create_sequences, build_lstm_model
import matplotlib.pyplot as plt

data, scaler = load_and_preprocess_data()
sequence_length = 12
features = data.drop(columns=['SalePrice']).values
target = data['SalePrice'].values
X, y = create_sequences(features, target, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

predicted_prices = model.predict(X_test)

predicted_prices_rescaled = scaler.inverse_transform(predicted_prices)
actual_prices_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(actual_prices_rescaled, predicted_prices_rescaled)
mse = mean_squared_error(actual_prices_rescaled, predicted_prices_rescaled)
r2 = r2_score(actual_prices_rescaled, predicted_prices_rescaled)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

plt.plot(actual_prices_rescaled, color='blue', label='Actual Prices')
plt.plot(predicted_prices_rescaled, color='red', label='Predicted Prices')
plt.title('House Price Prediction')
plt.xlabel('Time')
plt.ylabel('House Price')
plt.legend()
plt.show()
