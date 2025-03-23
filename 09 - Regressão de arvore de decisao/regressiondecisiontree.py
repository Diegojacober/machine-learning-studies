import preprocessing as pre
import numpy as np
import pandas as pd

# temporizador
import time
from functools import wraps


def computeDecisionTreeRegressionModel(X, y):
    from sklearn.tree import DecisionTreeRegressor

    regressor = DecisionTreeRegressor()
    regressor.fit(X, y)

    return regressor


def showPlot(XPoints, yPoints, XLine, yLine):
    import matplotlib.pyplot as plt

    plt.scatter(XPoints, yPoints, color='red')
    plt.plot(XLine, yLine, color='blue')
    plt.title(
        "Comparando pontos reais com a reta produzida pela regressão de árvore de decisão.")
    plt.xlabel("Experiência em anos")
    plt.ylabel("Salário")
    plt.show()


def runDecisionTreeRegressionExample(filename):
    start_time = time.time()
    X, y, csv = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    dtModel = computeDecisionTreeRegressionModel(X, y)
    elapsed_time = time.time() - start_time
    print("Compute Decision Tree Regression: %.2f" % elapsed_time, "segundos.")

    showPlot(X, y, X, dtModel.predict(X))

    XGrid = np.arange(min(X), max(X), 0.01)
    XGrid = XGrid.reshape((len(XGrid), 1))
    showPlot(X, y, XGrid, dtModel.predict(XGrid))

    from sklearn.tree import export_graphviz

    # export the decision tree to a tree.dot file
    # for visualizing the plot easily anywhere
    export_graphviz(dtModel, out_file='tree.dot',
                    feature_names=['Experiência'])


if __name__ == "__main__":
    runDecisionTreeRegressionExample("salary2.csv")
