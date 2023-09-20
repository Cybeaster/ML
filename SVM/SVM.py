from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

iris = load_iris(as_frame=True)

x = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)

svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))

svm_clf.fit(x, y)
x_new = [[5.5, 1.5], [5.0, 1.5]]
print(svm_clf.predict(x_new))
print(svm_clf.decision_function(x_new))

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(x, y)
