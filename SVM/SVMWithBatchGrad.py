import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)  # Iris virginica


class LinearSVC(BaseEstimator):
    def __init__(self, C=1, eta0=1, eta_d=10_000, n_epochs=1000, random_state=None):
        self.Js = None
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.eta_d = eta_d

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)

    def fit(self, x, y):
        # If a random state is set, use it to seed the random number generator
        if self.random_state:
            np.random.seed(self.random_state)

        # Initialize weight vector with random values and bias to 0
        w = np.random.randn(x.shape[1], 1)
        b = 0

        # Get the number of samples
        m = len(x)
        # Transform y to have values -1 and 1
        t = y * 2 - 1
        # Scale the features by the transformed target values
        x_t = x * t
        # Initialize a list to store the cost function values during training
        self.Js = []
        # Training loop: iterate over the specified number of epochs
        for epoch in range(self.n_epochs):
            # Identify the support vectors (points inside the margin)
            support_vector_idx = (x_t.dot(w) + t * b < 1).ravel()

            # Get the support vectors and their corresponding target values
            x_t_sv = x_t[support_vector_idx]
            t_sv = t[support_vector_idx]

            # Calculate the cost function using hinge loss and an L2 regularization term
            j = 1 / 2 * np.sum(w * w) + self.C * (np.sum(1 - x_t_sv.dot(w))) - b * np.sum(t_sv)

            # Store the current value of the cost function
            self.Js.append(j)

            w_gradient_vector = w - self.C * np.sum(x_t_sv, axis=0).reshape(-1, 1)
            b_gradient = -self.C * np.sum(t_sv)

            w = w - self.lr * w_gradient_vector
            b = b - self.lr * b_gradient

        self.w = w
        self.b = b
