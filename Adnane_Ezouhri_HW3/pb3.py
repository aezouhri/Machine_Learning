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



# import numpy as np
# import matplotlib.pyplot as plt

# # Data
# t_values = np.array([0.115, 0.116, 0.625])
# y_values = np.array([-1, -1, 1])
# X = np.column_stack((np.ones(t_values.shape), np.cos(2 * np.pi * t_values), np.sin(2 * np.pi * t_values)))

# # Ridge Regression Function
# def ridge_regression(X, y, lamb):
#     return np.linalg.inv(X.T @ X + lamb * np.eye(X.shape[1])) @ X.T @ y

# # Generate Lambda Values
# lambdas = np.linspace(0, 10, 20)

# # (c) Plotting the error as a function of lambda
# errors_lambda = []

# for lamb in lambdas:
#     theta_lambda = ridge_regression(X, y_values, lamb)
#     y_lambda = X @ theta_lambda
#     error = np.linalg.norm(y_lambda - y_values)**2
#     errors_lambda.append(error)

# # (d) Compute least-square coefficients for noisy instances
# num_instances = 10
# std_dev = 0.5
# errors_ls_noisy = []

# for _ in range(num_instances):
#     y_noisy = y_values + np.random.normal(0, std_dev, y_values.shape)
#     theta_ls_noisy = ridge_regression(X, y_noisy, 0)  # Lambda = 0 for LS
#     y_pred_noisy = X @ theta_ls_noisy
#     error = np.linalg.norm(y_pred_noisy - y_noisy)**2
#     errors_ls_noisy.append(error)

# avg_error_ls_noisy = sum(errors_ls_noisy) / num_instances

# # (e) Compute ridge regression coefficients for noisy instances
# errors_lambda_noisy = []

# for lamb in lambdas:
#     errors_for_lambda = []
    
#     for _ in range(num_instances):
#         y_noisy = y_values + np.random.normal(0, std_dev, y_values.shape)
#         theta_lambda_noisy = ridge_regression(X, y_noisy, lamb)
#         y_lambda_noisy = X @ theta_lambda_noisy
#         error = np.linalg.norm(y_lambda_noisy - y_noisy)**2
#         errors_for_lambda.append(error)

#     avg_error_for_lambda = sum(errors_for_lambda) / num_instances
#     errors_lambda_noisy.append(avg_error_for_lambda)

# # Plotting all the plots together
# plt.figure(figsize=(15, 10))

# # Plot for part (c)
# plt.subplot(2, 2, 1)
# plt.plot(lambdas, errors_lambda)
# plt.xlabel("Lambda")
# plt.ylabel("Error")
# plt.title("(c) Error as a function of Lambda")
# plt.grid(True)

# # Plot for part (d)
# plt.subplot(2, 2, 2)
# plt.plot(range(num_instances), errors_ls_noisy, label='Error in Each Noisy Instance')
# plt.axhline(y=avg_error_ls_noisy, color='r', linestyle='-', label='Average Error')
# plt.xlabel("Instance")
# plt.ylabel("Error")
# plt.title("(d) Errors in Noisy Instances with Least Squares")
# plt.legend()
# plt.grid(True)

# # Plot for part (e)
# plt.subplot(2, 2, 3)
# plt.plot(lambdas, errors_lambda_noisy)
# plt.xlabel("Lambda")
# plt.ylabel("Error")
# plt.title("(e) Average Error as a function of Lambda with Noisy Data")
# plt.grid(True)

# plt.tight_layout()
# plt.show()