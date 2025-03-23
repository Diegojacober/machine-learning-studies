import preprocessing as pre
import numpy as np
import pandas as pd

# temporizador
import time
from functools import wraps
from sklearn.svm import SVR


def computeSupportVectorRegressionModel(X, y, k, d):
    if (k == 'poly'):
        regressor = SVR(kernel=k, degree=d)
    else:
        regressor = SVR(kernel=k, gamma=1000.0)
    regressor.fit(X, np.ravel(y))

    return regressor


def showPlot(XPoints, yPoints, XLine, yLine):
    import matplotlib.pyplot as plt

    plt.scatter(XPoints, yPoints, color='red')
    plt.plot(XLine, yLine, color='blue')
    plt.title(
        "Comparando pontos reais com a reta produzida pela regressão de vetor suporte.")
    plt.xlabel("Experiência em anos")
    plt.ylabel("Salário")
    plt.show()


def runSupportVectorRegressionExample(filename):
    start_time = time.time()
    X, y, csv = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    X, scaleX = pre.computeScaling(X)
    y, scaleY = pre.computeScaling(np.reshape(y, shape=(-1, 1)))
    elapsed_time = time.time() - start_time
    print("Compute Scaling: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    svrModel = computeSupportVectorRegressionModel(X, y, "linear", 2)
    elapsed_time = time.time() - start_time
    print("Compute SVR with kernel Linear: %.2f" % elapsed_time, "segundos.")

    showPlot(
        scaleX.inverse_transform(X),
        scaleY.inverse_transform(y),
        scaleX.inverse_transform(X),
        scaleY.inverse_transform(svrModel.predict(X).reshape(-1, 1))
    )

    start_time = time.time()
    svrModel = computeSupportVectorRegressionModel(X, y, "poly", 3)
    elapsed_time = time.time() - start_time
    print("Compute SVR with kernel Poly: %.2f" % elapsed_time, "segundos.")

    showPlot(
        scaleX.inverse_transform(X),
        scaleY.inverse_transform(y),
        scaleX.inverse_transform(X),
        scaleY.inverse_transform(svrModel.predict(X).reshape(-1, 1))
    )

    start_time = time.time()
    svrModel = computeSupportVectorRegressionModel(X, y, "rbf", 2)
    elapsed_time = time.time() - start_time
    print("Compute SVR with kernel RBF: %.2f" % elapsed_time, "segundos.")

    showPlot(
        scaleX.inverse_transform(X),
        scaleY.inverse_transform(y),
        scaleX.inverse_transform(X),
        scaleY.inverse_transform(svrModel.predict(X).reshape(-1, 1))
    )


if __name__ == "__main__":
    runSupportVectorRegressionExample("salary.csv")
