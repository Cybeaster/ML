from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve

from Data import X_train_set, y_train_5, X
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train_set, y_train_5)
cross_val_score(sgd_clf, X_train_set, y_train_5, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(sgd_clf, X_train_set, y_train_5, cv=3)
conf_mat = confusion_matrix(y_train_5, y_train_pred)

y_train_perf_pred = y_train_5
confusion_matrix(y_train_5, y_train_perf_pred)

# calc precision/recall score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))\

# precision/recall tradeoff
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)


y_scores = cross_val_predict(sgd_clf, X_train_set, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

idx_for_90_precision = (precisions >= 90).argmax()
thresholds_for_90_precision = thresholds[idx_for_90_precision]

# instead of callin predict method we could call:
y_train_pred_90 = (y_scores >= thresholds_for_90_precision)

# get roc curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
