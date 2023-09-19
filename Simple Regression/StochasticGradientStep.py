import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import add_dummy_feature
from sklearn.linear_model import SGDRegressor

# manually build stochastic gradient steps and then show how it looks in sklearn

np.random.seed(42)
m = 100
n_epoch = 50
t0, t1 = 5, 50


def learning_schedule(t):
    return t0 / (t + t1)


x = 2 * np.random.rand(m, 1)  # column
y = 4 + 3 * x + np.random.randn(m, 1)  # column

X_b = add_dummy_feature(x)
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new)  # add x0 = 1 to each instance
n_shown = 50  # just needed to generate the figure below

theta = np.random.randn(2, 1)

for epoch in range(n_epoch):
    for iteration in range(m):

        # these 4 lines are used to generate the figure
        if epoch == 0 and iteration < n_shown:
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(iteration / n_shown + 0.15))
            plt.plot(X_new, y_predict, color=color)

        # randomly pick data, e.g. stochastic
        random_index = np.random.randint(m)
        xi = X_b[random_index: random_index + 1]
        yi = y[random_index: random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

# beautifies and saves
plt.plot(x, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.show()

# OR IN SKLEARN WAY

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01, n_iter_no_change=100, random_state=42)
sgd_reg.fit(x, y.ravel())
