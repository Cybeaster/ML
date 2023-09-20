from scipy.stats import loguniform, uniform
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

housing = fetch_california_housing(as_frame=True)

x = housing.data
y = housing.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

lin_svc = make_pipeline(StandardScaler(), LinearSVR(max_iter=5000, random_state=42))
lin_svc.fit(x_train, y_train)

y_pred = lin_svc.predict(x_train)
mse = mean_squared_error(y_train, y_pred)
print(mse)

# Do now with SVR
svm_clf = make_pipeline(StandardScaler, SVR())

param_distr = {
    "svr__gamma": loguniform(0.001, 0.1),
    "svr__C": uniform(1, 10)
}

rnd_search = RandomizedSearchCV(svm_clf, param_distr, n_iter=100, cv=3, random_state=42)
rnd_search.fit(x_train[:2000], y_train[:2000])
