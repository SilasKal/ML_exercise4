import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class OrdinaryLeastSquares:
    def __int__(self):
        self.beta = beta

    def fit(self, X, Y):
        if X.ndim > 1:  # if multiple independent variables
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)  # extend matrix with column of ones
            self.beta = np.dot((np.linalg.inv(np.dot(X.T, X))), np.dot(X.T, Y))  # calculate coefficient and intercept
        else:  # only one independent variable
            x_mean = np.mean(X)
            y_mean = np.mean(Y)
            numerator = np.sum((X - x_mean) * (Y - y_mean))  # formula from lecture slides
            denominator = np.sum(np.square(X - x_mean))
            b = numerator / denominator
            a = y_mean - b * x_mean
            # print(a, b)
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
plt.show()

# array2 = np.array([[156.3, 47.1], [158.9, 46.8], [160.8, 49.3], [179.6, 53.2], [156.6, 47.7], [165.1, 49.0], [165.9, 50.6], [156.7, 47.1], [167.8, 51.7], [160.8, 47.8]])
# OLS2 = OrdinaryLeastSquares()
# OLS2.fit(array2[:, 0], array2[:, 1])
# y2_predict = OLS2.predict(array2[:, 0])
# LRG = sklearn.linear_model.LinearRegression()
# LRG.fit(array2[:, 0].reshape(-1, 1), array2[:, 1])
# y2_predict_1 = LRG.predict(array2[:, 0].reshape(-1, 1))
# print(LRG.coef_, LRG.intercept_,  "lrg coef, intercept")
#
# plt.scatter(array2[:, 0], y2_predict, c= "blue")
# plt.scatter(array2[:, 0], y2_predict_1, c= "red")
# plt.show()
