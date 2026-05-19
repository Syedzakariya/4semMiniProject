import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import yfinance as yf

# Download stock data
data = yf.download('AAPL',
                   start='2020-01-01',
                   end='2024-01-01')

# Display first few rows
print(data.head())

# Select closing prices
closing_prices = data['Close'].values

# Reshape data
closing_prices = closing_prices.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(closing_prices)

# Create training data
x_train = []
y_train = []

for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

# Convert to arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape for LSTM
x_train = np.reshape(
    x_train,
    (x_train.shape[0],
     x_train.shape[1],
     1)
)

# Build the model
model = Sequential()

# First LSTM layer
model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(x_train.shape[1], 1)
    )
)

# Second LSTM layer
model.add(
    LSTM(
        units=50
    )
)

# Output layer
model.add(
    Dense(
        units=1
    )
)

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Train model
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32
)

# Predict prices
predictions = model.predict(x_train)

# Convert back to original values
predictions = scaler.inverse_transform(predictions)

real_prices = scaler.inverse_transform(
    y_train.reshape(-1, 1)
)

# Plot graph
plt.figure(figsize=(12,6))

plt.plot(
    real_prices,
    color='blue',
    label='Actual Stock Price'
)

plt.plot(
    predictions,
    color='red',
    label='Predicted Stock Price'
)

plt.title('Stock Market Prediction Using LSTM')

plt.xlabel('Time')

plt.ylabel('Stock Price')

plt.legend()

plt.show()
