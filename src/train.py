import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Create results folder
os.makedirs("results", exist_ok=True)

# Load data
data = pd.read_csv("data/flux_data.csv")

# Inputs (p1-p4)
X = data.iloc[:, 0:4].values

# Outputs (flux values)
y = data.iloc[:, 4:].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Error
mse = mean_squared_error(y_test, pred)
print("MSE:", mse)

# Plot
true_flux = y_test[0]
pred_flux = pred[0]

plt.plot(true_flux, label="True Flux")
plt.plot(pred_flux, label="Predicted Flux")
plt.legend()
plt.title("AI Flux Prediction")
plt.savefig("results/flux_plot.png")
plt.show()
