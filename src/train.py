import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Create results folder if it does not exist
os.makedirs("results", exist_ok=True)

# Dummy dataset to simulate neutron flux behaviour
np.random.seed(42)

n_samples = 200
n_features = 4
profile_length = 50

# Input parameters
X = np.random.rand(n_samples, n_features)

# Simulated neutron flux profiles
y = np.array([
    np.sin(np.linspace(0, 3, profile_length) + x[0] * 2) * (1 + x[1]) + 0.1 * x[2]
    for x in X
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Neural network model
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Error
mse = mean_squared_error(y_test, pred)
print("Test MSE:", mse)

# Plot comparison
true_flux = y_test[0]
pred_flux = pred[0]

plt.figure(figsize=(8, 5))
plt.plot(true_flux, label="True Flux")
plt.plot(pred_flux, label="Predicted Flux")
plt.xlabel("Position")
plt.ylabel("Neutron Flux")
plt.title("AI Prediction vs True Neutron Flux")
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig("results/flux_comparison.png", dpi=300)

# Show figure
plt.show()
