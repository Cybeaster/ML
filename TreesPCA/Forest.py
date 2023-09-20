import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import shuffle_split, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from scipy.stats import mode

n_trees = 1000
n_instances = 100

mini_sets = []

x, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rs = shuffle_split(n_splits=n_trees, test_size=len(x_train) - n_instances, random_state=42)

for mini_train_idx, mini_test_idx in rs.split(x_train):
    x_mini_train = x_train[mini_train_idx]
    y_mini_train = y_train[mini_train_idx]
    mini_sets.append((x_mini_train, y_mini_train))

params = {
    'max_leaf_nodes': list(range(2, 100)),
    'max_depth': list(range(1, 7)),
    'min_samples_split': [2, 3, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search.fit(x_train, y_train)

forest = [clone(grid_search.best_estimator_) for _ in range(n_trees)]
accuracy_scores = []

for tree, (x_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(x_mini_train, y_mini_train)

    y_pred = tree.predict(x_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(np.mean(accuracy_scores))
y_pred = np.empty([n_trees, len(x_test)], dtype=np.uint8)

for tree_idx, tree in enumerate(forest):
    y_pred[tree_idx] = tree.predict(x_test)

y_pred_majority_votes, n_votes = mode(y_pred, axis=0)
