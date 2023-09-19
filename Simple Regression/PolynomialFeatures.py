from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor

import numpy as np
#trying to learn from polynomial data
np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)  # make random linear data

x_train,y_train = X[: m // 2], y[: m // 2, 0]
x_valid,y_valid = X[m // 2 :], y[m // 2 :, 0]

preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler()) # better to scale parameters


x_train_prep = preprocessing.fit_transform(x_train)
x_valid_prep = preprocessing.transform(x_valid)