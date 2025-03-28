from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
chp2 = pd.get_dummies(chp, columns=['ocean_proximity'], drop_first=True)

# print(chp2.columns)

# remover o preço que vai ser previsto
chp2 = chp2[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
             'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity_INLAND',
             'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
             'ocean_proximity_NEAR OCEAN']]

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

    print(f"Limite Inferior: {limIn}, Limite Superior: {limSup}")

    # Marcar outliers como NaN
    data.loc[(data[column] < limIn) | (data[column] > limSup), column] = np.nan

    # Imputation for completing missing values using k-Nearest Neighbors. Each sample's missing values are imputed using the mean value from n_neighbors nearest
    imputer = KNNImputer(n_neighbors=15, weights='uniform',
                         metric='nan_euclidean')
    data[[column]] = imputer.fit_transform(data[[column]])

    print("Outliers tratados e valores imputados com sucesso.")
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


# KNNImputer: Melhor para manter a estrutura dos dados, ideal quando há relação entre variáveis.

# Remoção: Adequado quando há poucos outliers e muitos dados.

# Média, Mediana ou Moda: Simples e eficiente para pequenas quantidades de outliers.

# Winsorization: Bom para manter os dados originais, limitando o impacto dos outliers.

# Transformações: Ideal para dados com assimetria ou distribuição não normal.

# Regressão: Funciona bem em conjuntos de dados com forte correlação entre variáveis.

# chp2.hist(figsize=(16, 12), bins=50)
# plt.show()

# tratando os principais outliers
# plotOutliers(chp2, 'total_bedrooms')
identifyAndImputeOutliers(chp2, 'total_bedrooms')
identifyAndImputeOutliers(chp2, 'total_rooms')

# chp2.hist(figsize=(16, 12), bins=50)
# plt.show()

# print(chp2['total_bedrooms'].describe())
# print(chp2['total_bedrooms'].isna().sum())

# plt.figure(figsize=(8, 6))
# plt.hist(chp2['total_bedrooms'], bins=50, color='blue', edgecolor='black')
# plt.xlabel('Total Bedrooms')
# plt.ylabel('Frequency')
# plt.title('Distribution of Total Bedrooms')
# plt.show()

# chp2.boxplot(figsize=(16, 2))
# plt.show()

chp3 = chp2.dropna()
# print(chp3.describe())
# plt.figure(figsize=(16, 10))
# sns.heatmap(chp3.corr(), annot=True, cmap="seismic")
# plt.show()

# plt.figure(figsize=(16, 10))
# sns.scatterplot(x='latitude', y='longitude', data=chp3,
#                 hue='median_house_value')
# plt.show()

print(chp3.columns)

# plt.figure(figsize=(16, 10))
# sns.pairplot(chp3[['housing_median_age', 'total_rooms',
#                    'total_bedrooms', 'population', 'households', 'median_income',
#                    'median_house_value']])
# plt.show()

# region Machine Learning


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


# region Scaling
X = chp3.drop("median_house_value", axis=1)
y = chp3["median_house_value"]

# Exibindo as primeiras linhas de X e y sem escala
print(f"X sem scale:\n{X.head()}")
print(f"Y sem scale:\n{y.head()}")

# Padronizando X e y
X_scaled, scaleX = computeScaling(X)
y_scaled, scaleY = computeScaling(np.reshape(y, (-1, 1)))

# Exibindo as primeiras linhas de X e y após escala
# Mostrando apenas as primeiras 5 linhas para visualização
print(f"X com scale:\n{X_scaled[:5]}")
# Mostrando apenas as primeiras 5 linhas para visualização
print(f"Y com scale:\n{y_scaled[:5]}")

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


XTrain, XTest, yTrain, yTest = splitTrainTestSets(
    X_scaled, y_scaled, 0.3)  # Usando X_scaled e y_scaled

# Exibindo a forma dos dados de treino e teste
print(f"XTrain shape: {XTrain.shape}, XTest shape: {XTest.shape}")
print(f"yTrain shape: {yTrain.shape}, yTest shape: {yTest.shape}")

# endregion Train e test split

# region Linear Regression

# linear = LinearRegression()
# linear.fit(XTrain, yTrain)
# predicao_linear = linear.predict(XTest)

# # Desescalonando as previsões
# predicao_linear_desescalonada = scaleY.inverse_transform(predicao_linear)

# # Desescalonando os valores reais (yTest)
# yTest_desescalonado = scaleY.inverse_transform(yTest)

# # Calculando as métricas de erro com os valores desescalonados
# mae = mean_absolute_error(yTest_desescalonado, predicao_linear_desescalonada)
# mse = mean_squared_error(yTest_desescalonado, predicao_linear_desescalonada)
# rmse = np.sqrt(mse)

# # Exibindo as métricas
# print(f"MAE: {mae}")
# print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")

# # Avaliação do modelo
# print(f"R² (score): {linear.score(XTest, yTest)}")
# print(
#     f"R² (manual): {r2_score(yTest_desescalonado, predicao_linear_desescalonada)}")

# print(f"Previsões desescalonadas: {predicao_linear_desescalonada[:5]}")
# print(f"Valores reais desescalonados: {yTest_desescalonado[:5]}")

# endregion


# region Todos os modelos
# Definindo os modelos
modelos = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'Ridge': Ridge(),
    'BayesianRidge': BayesianRidge()
}

# Listas para armazenar os resultados
modelo_rmse = []
modelo_mae = []
modelo_mse = []
modelo_r2 = []
modelo_nomes = []
modelo_erro_percentual = []
modelo_acuracia = []

# Iterando sobre os modelos
# Iterando sobre os modelos
for nome, modelo in modelos.items():
    # Treinando o modelo
    # Usando .ravel() para garantir que y seja 1D
    fit_modelo = modelo.fit(XTrain, yTrain.ravel())

    # Fazendo previsões
    predicao = fit_modelo.predict(XTest)

    # Desescalonando as previsões e yTest
    predicao_desescalonada = scaleY.inverse_transform(
        predicao.reshape(-1, 1))  # reshape para garantir a forma correta
    yTest_desescalonado = scaleY.inverse_transform(yTest)

    # Calculando as métricas de erro
    mae = mean_absolute_error(yTest_desescalonado, predicao_desescalonada)
    mse = mean_squared_error(yTest_desescalonado, predicao_desescalonada)
    rmse = np.sqrt(mse)
    r2 = r2_score(yTest_desescalonado, predicao_desescalonada)

    # Calculando a porcentagem de erro
    erro_percentual = np.abs(
        (yTest_desescalonado - predicao_desescalonada) / yTest_desescalonado) * 100
    erro_percentual_medio = np.mean(erro_percentual)

    # Definindo um limite de erro de 10% para considerar uma previsão como "acertada"
    erro_limite = 10
    acertos = np.sum(erro_percentual <= erro_limite)
    porcentagem_acertos = (acertos / len(erro_percentual)) * 100

    # Armazenando os resultados
    modelo_r2.append(r2)
    modelo_rmse.append(rmse)
    modelo_mae.append(mae)
    modelo_mse.append(mse)
    modelo_nomes.append(nome)
    # Adicionando erro percentual
    modelo_erro_percentual.append(erro_percentual_medio)
    modelo_acuracia.append(porcentagem_acertos)  # Adicionando acurácia

# Criando o DataFrame para exibir os resultados
resultado_final = pd.DataFrame({
    'Modelo': modelo_nomes,
    'R2': modelo_r2,
    'MAE': modelo_mae,
    'MSE': modelo_mse,
    'RMSE': modelo_rmse,
    'Erro Percentual Médio': modelo_erro_percentual,  # Exibindo o erro percentual
    'Acurácia (%)': modelo_acuracia  # Exibindo a acurácia
})

# Ordenando pelo RMSE e exibindo os resultados
resultado_final = resultado_final.sort_values(by='RMSE')
print(resultado_final)

# endregion

# endregion

# endregion
