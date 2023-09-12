import tarfile
import urllib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def load_tit_data():
    path = Path("datasets/titanic.tgz")
    if not path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as tit_data:
            tit_data.extractall(path="datasets")

    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]


train_data, test_data = load_tit_data()
train_data = train_data.set_index("PassengerId")
test_data = train_data.set_index("PassengerId")
train_data.info()

num_pipeline = Pipeline([  # pipeline for nums
    ("imputer", SimpleImputer(strategy="median")),  # Give missing values median ones
    ("scaler", StandardScaler())  # scale all values using standart scaler
])

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

cat_pipeline = Pipeline([  # pipeline for categories
    ("ordinal_encoder", OrdinalEncoder()),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", OneHotEncoder(sparse=False))
])

# it's needed to tell what columns are what
num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocessing_pipe = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

x_train = preprocessing_pipe.fit_transform(train_data)
y_train = train_data["Survived"]

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(x_train, y_train)

x_test_data = preprocessing_pipe(test_data)
y_prediction = forest_clf.predict(x_test_data)

forest_score = cross_val_score(forest_clf, x_train, y_train, cv=10)
forest_score.mean()
