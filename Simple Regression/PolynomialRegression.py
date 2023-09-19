import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from math import ceil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import matplotlib as mpl

np.random.seed(0)

m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)  # y = ax^2 + bx + c

poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)

min_batch_size = 50
n_epochs = 50

epoch = 20
n_batches_per_epoch = ceil(m / min_batch_size)
theta = np.random.randn(3, 1)

t0, t1 = 200, 2000


def learning_schedule(t):
    return t0 / (t + t1)


theta_path_mgd = []

for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)

    x_shuffled = x_poly[shuffled_indices]
    y_shuffled = y[shuffled_indices]  # shuffle x and y

    for iteration in range(0, n_batches_per_epoch):
        idx = iteration * min_batch_size  # take indices for each batch
        xi = x_shuffled[idx: idx + min_batch_size]
        yi = y_shuffled[idx: idx + min_batch_size]
        gradients = 2 / min_batch_size * xi.T @ (xi @ theta - yi)  # take gradients for each

        eta = learning_schedule(iteration)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)  # remember theta

theta_path_mgd = np.array(theta_path_mgd)

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

X_new = np.linspace(-3, 3, 100).reshape(100, 1)  # take new data (interval)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.figure(figsize=(6, 4))
plt.plot(x, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.legend(loc="upper left")
plt.axis([-3, 3, 0, 10])
plt.grid()
plt.show()

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection='3d')

ax.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], theta_path_mgd[:, 2], "g-+", linewidth=2,
        label="Mini-batch")
ax.legend(loc="upper left")

ax.set_xlabel(r"$\theta_0$")
ax.set_ylabel(r"$\theta_1$")
ax.set_zlabel(r"$\theta_2$")

theta_stoch = np.random.randn(3, 1)  # 3*1 array

theta_path_stch = []
n_shown = 20

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = x_poly[random_index: random_index + 1]
        yi = y[random_index: random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta_stoch - yi)

        eta = learning_schedule(epoch * m + iteration)  # take a specific eta for this learning step
        theta_stoch = theta_stoch - eta * gradients
        theta_path_stch.append(theta_stoch)

theta_path_stch = np.array(theta_path_stch)
ax.plot(theta_path_stch[:, 0], theta_path_stch[:, 1], theta_path_stch[:, 2], "r--", linewidth=2, label="Stochastic")
ax.legend()
plt.show()
