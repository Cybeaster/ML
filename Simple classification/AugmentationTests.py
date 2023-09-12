
from scipy.ndimage import shift
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import DataSet

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

kn_clf = KNeighborsClassifier()
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6]}]
kn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(kn_clf,param_grid=param_grid,cv=5)
grid_search.fit(DataSet.X_train_set[:10_000],DataSet.Y_train_set[:10_000])

grid_search.best_estimator_.fit(DataSet.X_train_set,DataSet.Y_train_set)

x_train_augment = [image for image in DataSet.X_train_set]
y_train_augment = [label for label in DataSet.Y_train_set]

print("start augmentation!")

for dy,dx in ((-1,0),(1,0),(0,1),(0,-1)):
    for image,label in zip(DataSet.X_train_set,DataSet.Y_train_set):
        x_train_augment.append(shift_image(image,dx,dy))
        y_train_augment.append(label)

print("end augmentation!")

x_train_augment = np.array(x_train_augment)
y_train_augment = np.array(y_train_augment)

shuffl_idx = np.random.permutation(len(x_train_augment))

x_train_augment = x_train_augment[shuffl_idx]
y_train_augment = y_train_augment[shuffl_idx]

print("start training")
augmented_kn_clf = KNeighborsClassifier(**grid_search.best_params_)
augmented_kn_clf.fit(x_train_augment[:60000 * 4],y_train_augment[:60000 * 4])

print(augmented_kn_clf.score(x_train_augment[60000 * 4:],y_train_augment[60000 * 4:]))

