from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
from Data import X_train_set,y_train_5
forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train_set, y_train_5, cv=3, method="predict_proba")

y_score_forest = y_probas_forest[:, 1]
precision_forest, recall_forest, threshold_forest = precision_recall_curve(y_train_5, y_score_forest)
