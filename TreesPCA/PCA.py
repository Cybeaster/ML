from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

x_iris, y_iris = load_iris(return_X_y=True)
pca_pipeline = make_pipeline(StandardScaler(), PCA())
x_iris_rot = pca_pipeline.fit_transform(x_iris)

tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(x_iris_rot, y_iris)
