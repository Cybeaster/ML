from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from Data import X_train_set,Y_train_set,X_test_set,Y_test_set
kn_clf = KNeighborsClassifier()

#first try
kn_clf.fit(X_train_set,Y_train_set)
accuracy = kn_clf.score(X_train_set,Y_test_set)


param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6]}]
kn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(kn_clf,param_grid=param_grid,cv=5)
grid_search.fit(X_train_set[:10_000],Y_train_set[:10_000])

grid_search.best_estimator_.fit(X_train_set,Y_train_set)

print(grid_search.score(X_test_set,Y_test_set))