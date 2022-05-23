import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn import datasets
from sklearn import linear_model

class OrdinaryLeastSquares:
    def __int__(self):
        self.beta = beta
    def fit(self, X, Y):
        if X.ndim > 1:
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
            self.beta = np.dot((np.linalg.inv(np.dot(X.T, X))), np.dot(X.T, Y))
        else:
            x_mean = np.mean(X)
            y_mean = np.mean(Y)
            numerator = np.sum((X-x_mean)*(Y- y_mean))
            denominator = np.sum(np.square(X-x_mean))
            b = numerator/denominator
            a = y_mean - b * x_mean
            # print(a, b)
            self.beta = (a, b)
    def predict(self, X):
        if X.ndim > 1:
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
            # print(self.beta)
            return np.dot(X, self.beta)
        else:
            # print(self.beta, "self beta")
            return self.beta[0] + self.beta[1]*X



iris = datasets.load_iris().data
training = iris[:100, :]
test = iris[100:, :]
X_test = test[:, :2]
Y_test = test[:, 2:]

X_training = training[:, :2]
Y_training = training[:, 2:]

OLS = OrdinaryLeastSquares()
OLS.fit(X_training, Y_training)
Y_predict = OLS.predict(X_test)
plt.scatter(test[:, 2],test[:, 3], c= "red")
plt.scatter(Y_predict[:, 0], Y_predict[:, 1], c= "blue")
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
