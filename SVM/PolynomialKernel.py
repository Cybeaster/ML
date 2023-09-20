from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm,clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

iris = load_iris(as_frame=True)

x = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)
poly_kernel_svm_clf.fit(x, y)

# Gaussian RBF kernel

rbf_kernel_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf"), SVC(kernel="rbf", gamma=5, C=0.001)
])
