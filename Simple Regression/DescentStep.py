import numpy as np
from sklearn.preprocessing import add_dummy_feature
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(42)

m = 100 #instances
x = 2 * np.random.rand(m,1) #column
y = 4 + 3 * x + np.random.randn(m,1) #column

X_b = add_dummy_feature(x)
eta = 0.1 #learning rate

n_epoch = 1000
m = len(X_b)
theta = np.random.randn(2,1) #randomly init model params
x_new = np.array([[0],[2]])
X_new_b = add_dummy_feature(x_new)

def plot_gradient_descent(theta, eta):
    m = len(X_b)
    plt.plot(x, y, "b.")
    n_epochs = 1000
    n_shown = 20
    theta_path = []
    for epoch in range(n_epochs):
        if epoch < n_shown:
            y_predict = X_new_b @ theta # computes possible y coord
            color = mpl.colors.rgb2hex(plt.cm.OrRd(epoch / n_shown + 0.15))
            plt.plot(x_new, y_predict, linestyle="solid", color=color)
        gradients = 2 / m * X_b.T @ (X_b @ theta - y) # takes new gradient with derivative of MSE
        theta = theta - eta * gradients # makes theta more precise
        theta_path.append(theta)
    plt.xlabel("$x_1$")
    plt.axis([0, 2, 0, 15])
    plt.grid()
    plt.title(fr"$\eta = {eta}$")
    return theta_path

plt.figure(figsize=(10, 4))

plt.subplot(131)
plot_gradient_descent(theta, eta=0.01)
plt.ylabel("$y$", rotation=0)

plt.subplot(132)
theta_path_bgd = plot_gradient_descent(theta, eta=0.1)
plt.gca().axes.yaxis.set_ticklabels([])

plt.subplot(133)
plt.gca().axes.yaxis.set_ticklabels([])
plot_gradient_descent(theta, eta=0.5)
plt.show()