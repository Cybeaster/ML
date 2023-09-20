from scipy.stats import loguniform, uniform
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

wine = load_wine(as_frame=True)

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=42)

lin_clf = make_pipeline(StandardScaler(), LinearSVC(random_state=42))
lin_clf.fit(x_train, y_train)

print("lin clf:", cross_val_score(lin_clf, x_train, y_train).mean())

param_distr = {
    "svc__gamma": loguniform(0.001, 0.1),
    "svc__C": uniform(1, 10)
}

svc_clf = make_pipeline(StandardScaler(), SVC(random_state=42))
print("svc clf:", cross_val_score(svc_clf, x_train, y_train).mean())

rnd_search = RandomizedSearchCV(svc_clf, param_distr, n_iter=100, cv=5, random_state=42)
rnd_search.fit(x_train, y_train)

print(rnd_search.best_estimator_)
print(rnd_search.score(x_test, y_test))
