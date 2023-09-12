from sklearn.svm import SVC
from Data import X_train_set, y_train_5, X, Y_train_set, some_digit

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train_set[:2000], Y_train_set[:2000])

somde_digit_score = svm_clf.decision_function([some_digit])
somde_digit_score.round(2)

class_id = somde_digit_score.argmax()