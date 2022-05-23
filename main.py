import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class OrdinaryLeastSquares:
    def __int__(self):
        self.beta = beta

    # fits model parameters to given data using least squares
    # when using one independent variable and multiple dependent variables use X.reshape(-1, 1)
    # in other cases use unshaped matrices
    def fit(self, X, Y):
        if X.ndim > 1:  # if multiple independent variables
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)  # extend matrix with column of ones
            self.beta = np.dot((np.linalg.inv(np.dot(X.transpose(), X))), np.dot(X.transpose(), Y))  # calculate coefficient and intercept
        else:  # only one independent variable
            x_mean = np.mean(X)
            y_mean = np.mean(Y)
            numerator = np.sum((X - x_mean) * (Y - y_mean))  # formula from lecture slides
            denominator = np.sum(np.square(X - x_mean))
            b = numerator / denominator
            a = y_mean - b * x_mean
            self.beta = (a, b)

    def predict(self, X):
        if X.ndim > 1:  # multiple independent variable
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)  # extend matrix with column of ones
            return np.dot(X, self.beta)  # calculate solution matrix
        else:  # one independent variable
            return self.beta[0] + self.beta[1] * X  # calculate a*x + b


iris = datasets.load_iris().data
training = iris[:100, :]  # first 100 samples as training data
test = iris[100:, :]  # remaining 50 samples as test data
X_test = test[:, :2]  # sepal length and sepal width as independent variables
Y_test = test[:, 2:]  # pedal length and pedal length as dependent variables

X_training = training[:, :2]  # sepal length and sepal width as independent variables
Y_training = training[:, 2:]  # pedal length and pedal length as dependent variables

OLS = OrdinaryLeastSquares()
OLS.fit(X_training, Y_training)
Y_predict = OLS.predict(X_test)
plt.scatter(test[:, 2], test[:, 3], c="red")  # plotting test data
plt.scatter(Y_predict[:, 0], Y_predict[:, 1], c="blue")  # plotting predicted data
plt.title("Predicting Petal Length and Width using Linear Regression")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
