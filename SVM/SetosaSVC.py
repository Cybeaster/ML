from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

iris = datasets.load_iris(as_frame=True)
x = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target

setosa_or_versicolor = (y == 0) | (y == 1)

x = x[setosa_or_versicolor]
y = y[setosa_or_versicolor]

C = 5

alpha = 0.05

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

linear_svc = LinearSVC(loss="hinge", C=C, random_state=42).fit(x_scaled, y)
svc = SVC(kernel="linear", C=C).fit(x_scaled, y)
sgd = SGDClassifier(alpha=alpha, random_state=42).fit(x_scaled, y)
