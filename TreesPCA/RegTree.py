from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

x_moons, y_moons = make_moons(n_samples=500, noise=0.2, random_state=42)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf_2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)

tree_clf.fit(x_moons, y_moons)
tree_clf_2.fit(x_moons, y_moons)

x_moon_test, y_moon_test = make_moons(n_samples=100, noise=0.2, random_state=43)

print(tree_clf.score(x_moon_test, y_moon_test))
print(tree_clf_2.score(x_moon_test, y_moon_test))
