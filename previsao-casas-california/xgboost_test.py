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

# explorando o dataset

# print(chp.info())
# print(chp.isnull().sum())

# print(chp['ocean_proximity'].unique())
# print(chp['ocean_proximity'].value_counts())

# transformar em colunas binárias
# print(pd.get_dummies(chp['ocean_proximity']))

# print(chp.drop('ocean_proximity', axis=1))
# chp2 = pd.get_dummies(chp, columns=['ocean_proximity'], drop_first=True)

# print(chp2.columns)

# remover o preço que vai ser previsto
chp2 = chp[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value',
            'ocean_proximity']]

# print(chp2.columns)

# tratar os valores que estão faltando
# print((207 / len(chp2)) * 100)
# equivale 1.002906976744186% da nossa base
# sns.displot(chp2['total_bedrooms'])
# plt.show()

# sns.boxplot(chp2['total_bedrooms'])
# plt.show()

# print(chp2['total_bedrooms'].mean())
# print(chp2['total_bedrooms'].median())


def identifyOutlier1(chp2):
    # identificando os outliers: baseando-se nos limites máximo e mínimo. Os limites são definidos pela média mais 3 vezes o desvio padrão como sendo o limite máximo e a média menos 3 vezes o desvio padrão sendo o limite mínimo.
    dt_mean, dt_std = np.mean(chp2['total_bedrooms']
                              ), np.std(chp2['total_bedrooms'])
    cut_off = dt_std * 3
    lower, upper = dt_mean - cut_off, dt_mean + cut_off
    ix = np.where((chp2['total_bedrooms'] < lower) |
                  (chp2['total_bedrooms'] > upper))

    chp2['Outlier'] = (chp2['total_bedrooms'] < lower) | (
        chp2['total_bedrooms'] > upper)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Gráfico 1: Dispersão completa
    axs[0].scatter(chp2.index, chp2['total_bedrooms'],
                   c=np.where(chp2['Outlier'], 'blue', 'red'))
    axs[0].set_title('Gráfico Completo')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('total_bedrooms')
    axs[0].legend(['Outliers', 'Normais'])

    # Gráfico 2: Dispersão RM x RM (exemplo apenas)
    axs[1].scatter(chp2['total_bedrooms'], chp2['total_bedrooms'],
                   c=np.where(chp2['Outlier'], 'blue', 'red'))
    axs[1].set_title('Gráfico Comparativo')
    axs[1].set_xlabel('total_bedrooms')
    axs[1].set_ylabel('total_bedrooms')
    axs[1].legend(['Outliers', 'Normais'])

    plt.suptitle('Método do Desvio')
    plt.tight_layout()
    plt.show()


def identifyAndImputeOutliers(data: pd.DataFrame, column):
    """
    Identifica e trata outliers usando o método de Tukey e aplica imputação com KNNImputer.
    O método de Tukey ou bloxplot consiste em definir os limites inferior e superior a partir do interquartil (IQR) e dos primeiros (Q1) e terceiros (Q3) quartis.

    Mas o que são quartis?

    Quartis são separatrizes que que dividem um conjunto de dados em 4 partes iguais. O objetivo das separatrizes é proporcionar uma melhor ideia da dispersão do conjunto de dados, principalmente da simetria ou assimetria da distribuição.

    O limite inferior é definido pelo primeiro quartil menos o produto entre o valor 1.5 e o interquartil.

    𝐿𝑖𝑛𝑓 = 𝑄1 − (1.5 ∗ 𝐼𝑄𝑅)

    O limite superior é definido pelo terceiro quartil mais o produto entre o valor 1.5 e o interquartil.

    𝐿𝑠𝑢𝑝 = 𝑄3 + (1.5 ∗ 𝐼𝑄𝑅)
    """
    # Calcula Q1, Q3 e IQR
    Q3 = data[column].quantile(0.75)
    Q1 = data[column].quantile(0.25)
    IQR = Q3 - Q1

    # Definir limites
    limIn = Q1 - (IQR * 1.5)
    limSup = Q3 + (IQR * 1.5)

    # print(f"Limite Inferior: {limIn}, Limite Superior: {limSup}")

    # Marcar outliers como NaN
    data.loc[(data[column] < limIn) | (data[column] > limSup), column] = np.nan

    # Imputation for completing missing values using k-Nearest Neighbors. Each sample's missing values are imputed using the mean value from n_neighbors nearest
    imputer = KNNImputer(n_neighbors=15, weights='uniform',
                         metric='nan_euclidean')
    data[[column]] = imputer.fit_transform(data[[column]])

    # print("Outliers tratados e valores imputados com sucesso.")
    # return data


def plotOutliers(data: pd.DataFrame, column: str):
    """
    Plota gráficos para visualizar os outliers antes e depois do tratamento.
    """
    data_before = data.copy()

    Q3 = data[column].quantile(0.75)
    Q1 = data[column].quantile(0.25)
    IQR = Q3 - Q1
    limIn = Q1 - (IQR * 1.5)
    limSup = Q3 + (IQR * 1.5)

    data_before['Outlier'] = (data_before[column] < limIn) | (
        data_before[column] > limSup)

    identifyAndImputeOutliers(data, column)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico Antes do Tratamento
    axs[0].scatter(data_before.index, data_before[column],
                   c=data_before['Outlier'].map({True: 'red', False: 'blue'}))
    axs[0].set_title(f'Antes do Tratamento ({column})')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel(column)

    # Gráfico Depois do Tratamento
    axs[1].scatter(data.index, data[column], color='blue')
    axs[1].set_title(f'Depois do Tratamento ({column})')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel(column)

    plt.suptitle('Visualização dos Outliers - Antes e Depois')
    plt.tight_layout()
    plt.show()


def winsorizeOutliers(data: pd.DataFrame, column: str, limit=0.05):
    data[column] = winsorize(data[column], limits=[limit, limit])
    return data


# plotOutliers(chp2, 'total_bedrooms')
identifyAndImputeOutliers(chp2, 'total_bedrooms')
identifyAndImputeOutliers(chp2, 'total_rooms')


def computeScaling(X) -> Tuple[np.ndarray, StandardScaler]:
    """
    Padroniza os dados de entrada X utilizando o StandardScaler. 
    Retorna os dados transformados e o scaler utilizado para a transformação.

    Parâmetros:
    X (np.ndarray): Dados de entrada a serem padronizados.

    Retorna:
    Tuple[np.ndarray, StandardScaler]: Os dados padronizados e o scaler.
    """
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)
    return X_scaled, scale


X = chp.drop("median_house_value", axis=1)
y = chp["median_house_value"]


# endregion Scaling

# region Train e test split


def splitTrainTestSets(X: np.ndarray, y: np.ndarray, testSize: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


houses = X.select_dtypes(exclude=np.number).columns.tolist()

for col in houses:
    X[col] = X[col].astype('category')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# Define hyperparameters
params = {"objective": "reg:squarederror", "device": "cuda"}
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

n = 200000
model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals,
    verbose_eval=50,
    # Activate early stopping
    early_stopping_rounds=50,
)


preds = model.predict(dtest_reg)

rmse = root_mean_squared_error(y_test, preds)

mae = mean_absolute_error(y_test, preds)
print(f"RMSE of the base model: {rmse:.3f}")
print(f"MAE of the base model: {mae:.3f}")


# results = xgb.cv(
#     params, dtrain_reg,
#     num_boost_round=n,
#     nfold=5,
#     early_stopping_rounds=20
# )

# best_rmse = results['test-rmse-mean'].min()

# print(best_rmse)
