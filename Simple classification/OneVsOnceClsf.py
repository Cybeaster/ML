import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC

from Data import X_train_set, Y_train_set
ovr_clf = OneVsOneClassifier(SVC(random_state=42))
ovr_clf.fit(X_train_set[:2000], Y_train_set[:2000])

y_train_large = (Y_train_set >= '7')
y_train_odd = (Y_train_set.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)

chain_clf.fit(X_train_set[:2000], y_multilabel)