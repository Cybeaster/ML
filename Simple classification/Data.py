import ssl

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

ssl._create_default_https_context = ssl._create_unverified_context

mnist = fetch_openml('mnist_784', as_frame=False)
def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


X, y = mnist.data, mnist.target

X_train_set, X_test_set, Y_train_set, Y_test_set = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (Y_train_set == '5')
y_test_5 = (Y_test_set == '5')
some_digit = X[0]
