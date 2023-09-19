import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import add_dummy_feature

np.random.seed(42)

m = 100  # instances
x = 2 * np.random.rand(m, 1)  # column
y = 4 + 3 * x + np.random.randn(m, 1)  # column

x_b = add_dummy_feature(x)

# calc theta to minimize cost function
theta_best = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
print(theta_best)

x_new = np.array([[0], [2]])
x_new_b = add_dummy_feature(x_new)
y_predict = x_new_b @ theta_best
print("y prediction is:", y_predict)

plt.plot(x_new, y_predict, "r-", label="Predictions")
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.plot(x, y, "b.")
plt.show()

theta_best_svd, residuals, rank, s = np.linalg.lstsq(x_b, y, rcond=1e-6)
custom_best_theta = np.linalg.pinv(x_b) @ y
