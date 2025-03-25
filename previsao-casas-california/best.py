from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from typing import Tuple
from scipy.stats.mstats import winsorize
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
sns.set_style('whitegrid')

chp = pd.read_csv("housing.csv")

chp['latitude_longitude'] = chp['latitude'] * chp['longitude']

chp2 = pd.get_dummies(chp, columns=['ocean_proximity'], drop_first=True)
chp2 = chp2[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
             'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity_INLAND',
             'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
             'ocean_proximity_NEAR OCEAN']]


def winsorizeOutliers(data: pd.DataFrame, column: str, limit=0.05):
    data[column] = winsorize(data[column], limits=[limit, limit])
    return data


chp2 = winsorizeOutliers(chp2, 'total_bedrooms', limit=0.1)


def computeScaling(X):
    scale = RobustScaler()
    X_scaled = scale.fit_transform(X)
    return X_scaled, scale


def splitTrainTestSets(X, y, testSize: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados em conjuntos de treino e teste.

    Parâmetros:
    X (np.ndarray): Dados de entrada (características).
    y (np.ndarray): Rótulos de saída.
    testSize (float): Proporção do conjunto de dados que será usado para teste.

    Retorna:
    Tuple: Dados de treino e teste para X e y.
    """
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=testSize)
    return XTrain, XTest, yTrain, yTest


X = chp2.drop("median_house_value", axis=1)
y = chp2["median_house_value"]
X_scaled, scaleX = computeScaling(X)
y_scaled, scaleY = computeScaling(np.reshape(y, (-1, 1)))
XTrain, XTest, yTrain, yTest = splitTrainTestSets(
    X_scaled, y_scaled, 0.3)

rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model.fit(XTrain, yTrain.ravel())
rf_preds = rf_model.predict(XTest)

rf_model_unscalled = scaleY.inverse_transform(rf_preds.reshape(-1, 1))
yTest_desescalonado = scaleY.inverse_transform(yTest)
mae = mean_absolute_error(yTest_desescalonado, rf_model_unscalled)
mse = mean_squared_error(yTest_desescalonado, rf_model_unscalled)
rmse = np.sqrt(mse)

# Exibindo as métricas
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


# XTrain, XTest, yTrain, yTest = splitTrainTestSets(
#     X, y, 0.25)

# model = ElasticNet(alpha=1.0, l1_ratio=0.5)
# model.fit(XTrain, yTrain)
# preds = model.predict(XTest)
# rmse = mean_squared_error(yTest, preds)
# print(f"RMSE after ElasticNet: {rmse:.3f}")
