from copy import deepcopy

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from SimpleRegression.PolynomialFeatures import x_train_prep, x_valid_prep, y_valid

# Try stochastic gradient descent

sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epoch = 500
best_valid_mse = float('inf')
train_errors, val_errors = [], []

for epoch in range(n_epoch):
    sgd_reg.partial_fit(x_train_prep, y_train)
    y_valid_pred = sgd_reg.predict(x_valid_prep)
    val_error = mean_squared_error(y_valid, y_valid_pred, squared=False)
    if val_error < best_valid_mse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)

    y_train_pred = sgd_reg.predict(x_train_prep)
    train_error = mean_squared_error(y_train, y_train_pred, squared=False)
    val_errors.append(val_error)
    train_errors.append(train_error)


best_epoch = np.argmin(val_errors)

plt.figure(figsize=(6, 4))
plt.annotate('Best model',
             xy=(best_epoch, best_valid_mse),
             xytext=(best_epoch, best_valid_mse + 0.5),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot([0, n_epoch], [best_valid_mse, best_valid_mse], "k:", linewidth=2)
plt.plot(val_errors, "b-", linewidth=3, label="Validation set")
plt.plot(best_epoch, best_valid_mse, "bo")
plt.plot(train_errors, "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.axis([0, n_epoch, 0, 3.5])
plt.grid()
plt.show()