import numpy as np
from Data import X_train_set, X_test_set

noise = np.random.randint(0, 100, (len(X_train_set), 784))
X_train_mod = X_train_set + noise
noise = np.random.randint(0, 100, (len(X_test_set), 784))

X_test_mod = X_test_set + noise
y_train_mod = X_train_set
y_test_mod = X_test_set
