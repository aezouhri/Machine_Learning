import numpy as np
import matplotlib.pyplot as plt

# Data
t_values = np.array([0.115, 0.116, 0.625])
y_values = np.array([-1, -1, 1])
X = np.column_stack((np.ones(t_values.shape), np.cos(2 * np.pi * t_values), np.sin(2 * np.pi * t_values)))

# Ridge Regression Function
def ridge_regression(X, y, lamb):
    return np.linalg.inv(X.T @ X + lamb * np.eye(X.shape[1])) @ X.T @ y

# Generate Lambda Values
lambdas = np.linspace(0, 10, 20)

# Part (c)
errors_c = []
for lamb in lambdas:
    theta_lambda = ridge_regression(X, y_values, lamb)
    y_lambda = X @ theta_lambda
    error = np.linalg.norm(y_lambda - y_values)**2
    errors_c.append(error)

# Part (d)
num_instances = 10
noise_stddev = 0.5
errors_d = []

for lamb in lambdas:
    errors_instance = []
    for _ in range(num_instances):
        noisy_y = y_values + np.random.normal(0, noise_stddev, y_values.shape)
        theta_noisy = ridge_regression(X, noisy_y, lamb)
        y_noisy_pred = X @ theta_noisy
        error = np.linalg.norm(y_noisy_pred - noisy_y)**2
        errors_instance.append(error)
    errors_d.append(np.mean(errors_instance))

# Part (e)
errors_e = []

for lamb in lambdas:
    errors_instance = []
    for _ in range(num_instances):
        noisy_y = y_values + np.random.normal(0, noise_stddev, y_values.shape)
        theta_ridge_noisy = ridge_regression(X, noisy_y, lamb)
        y_ridge_noisy_pred = X @ theta_ridge_noisy
        error = np.linalg.norm(y_ridge_noisy_pred - y_values)**2
        errors_instance.append(error)
    errors_e.append(np.mean(errors_instance))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(lambdas, errors_c, label='Part (c)')
plt.plot(lambdas, errors_d, label='Part (d)')
plt.plot(lambdas, errors_e, label='Part (e)')
plt.xlabel("Lambda")
plt.ylabel("Error")
plt.title("Error as a function of Lambda")
plt.legend()
plt.grid(True)
plt.show()
