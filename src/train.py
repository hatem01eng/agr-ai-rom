import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Dummy dataset to simulate neutron flux behaviour
np.random.seed(42)

n_samples = 200
n_features = 4
profile_length = 50

# Input parameters
X = np.random.rand(n_samples, n_features)

# Simulated neutron flux profiles
y = np.array([
    np.sin(np.linspace(0, 3, profile_length) + x[0]*2) * (1 + x[1]) + 0.1*x[2]
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

# Error calculation
mse = mean_squared_error(y_test, pred)

print("Test MSE:", mse)
