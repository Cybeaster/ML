import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

np.random.seed(42)

m = 100  # instances
x = 2 * np.random.rand(m, 1)  # column
y = 4 + 3 * x + np.random.randn(m, 1)  # column

lasso_reg = Lasso(alpha=0.1)  # SGDRegressor(penalty="l1", alpha=0.1).
lasso_reg.fit(x, y)
res = lasso_reg.predict([1.5])

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # combined
elastic_net.fit(x, y)
elastic_net.predict([[1.5]])
