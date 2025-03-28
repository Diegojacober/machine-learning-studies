import preprocessing as pre
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# temporizador
import time
from functools import wraps


def computePolynomialLinearRegressionModel(X, y, d):
    polynomialFeatures = PolynomialFeatures(degree=d)
    XPoly = polynomialFeatures.fit_transform(X)

    polyLinearRegression = LinearRegression()
    polyLinearRegression.fit(XPoly, y)

    return XPoly, polyLinearRegression


def showPlot(XPoints, yPoints, XLine, yLine):
    import matplotlib.pyplot as plt

    plt.scatter(XPoints, yPoints, color='red')  # plot real y points
    plt.plot(XLine, yLine, color='blue')  # plot predicted points in line
    plt.title(
        "Comparando pontos reais com a reta produzida pela regressão polinomial")
    plt.xlabel("Experiência em anos")
    plt.ylabel("Salário")
    plt.show()


def runPolynomialLinearRegressionExample(filename):
    start_time = time.time()
    X, y, csv = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    X = pre.fillMissingData(X, 0, 1)
    elapsed_time = time.time() - start_time
    print("Fill Missing Data: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    computePolynomialLinearRegressionModel(X, y, 4)
    elapsed_time = time.time() - start_time
    print("Compute Polynomial Linear Regression: %.2f" %
          elapsed_time, "segundos.")


if __name__ == "__main__":
    runPolynomialLinearRegressionExample("salary.csv")
