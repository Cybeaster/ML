from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

polynom_reg = make_pipeline(PolynomialFeatures(degree=10, include_bias=False), LinearRegression())

m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)  # y = ax^2 + bx + c

train_size, train_scores, valid_score = learning_curve(
    polynom_reg,
    x, y,
    train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5,
    scoring='neg_root_mean_squared_error')  # try learning curves to estimate the model

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_score.mean(axis=1)

plt.figure(figsize=(6, 4))  # not needed, just formatting
plt.plot(train_size, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_size, valid_errors, "b-", linewidth=3, label="valid")

# beautifies and saves Figure 4–15
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.grid()
plt.legend(loc="upper right")
plt.axis([0, 80, 0, 2.5])
plt.show()
