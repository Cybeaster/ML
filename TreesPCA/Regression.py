import numpy as np
from graphviz import Source
from sklearn.tree import DecisionTreeRegressor, export_graphviz

np.random.seed(42)

x_quad = np.random.rand(200, 1) - 0.5
y_quad = x_quad ** 2 + 0.025 * np.random.randn(200, 1)

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(x_quad, y_quad)

export_graphviz(tree_reg, out_file="regression_test.dot", feature_names=["x1"],
                rounded=True, filled=True)

source = Source.from_file("regression_test.dot")
source.view()
