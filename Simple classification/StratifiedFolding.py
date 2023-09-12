from sklearn import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from SGDClassification import X_train_set, y_train_5, sgd_clf, X


skfolds = StratifiedKFold(n_splits=3, shuffle=True)
for train_index, test_index in skfolds.split(X_train_set, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train_set[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train_set[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


