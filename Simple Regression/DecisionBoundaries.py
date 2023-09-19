import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
list(iris)


x = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train,y_train)

x_new = np.linspace(0,3,1000).reshape(-1,1)