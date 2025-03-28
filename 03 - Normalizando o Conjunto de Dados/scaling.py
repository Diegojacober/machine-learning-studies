from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

print("Carregando a base de dados...")
baseDeDados = pd.read_csv('admission.csv', delimiter=';')
X = baseDeDados.iloc[:, :-1].values
y = baseDeDados.iloc[:, -1].values
print("ok!")

print("Preenchendo dados que estão faltando...")
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit_transform(X[:, 1:])
print("ok!")

print("Computando rotulação...")
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

X = X[:, 1:]
D = pd.get_dummies(X[:, 0])
X = np.insert(X, 0, D.values, axis=1)
print("ok!")

print("Separando conjuntos de teste e treino...")
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
print("ok!")

# remover warning de dataconversionwarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# falar de distancia euclidiana pra justificar normalização
print("Computando normalização...")
scale_X = StandardScaler()
XTrain = scale_X.fit_transform(XTrain)
XTest = scale_X.fit_transform(XTest)
print("ok!")

print(yTest)
print(XTest)
