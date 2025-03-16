import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

baseDeDados = pd.read_csv('admission.csv', delimiter=';')
# geralmente usamos X para variaveis independentes e Y para as dependentes
X = baseDeDados.iloc[:, :-1].values
Y = baseDeDados.iloc[:, -1].values


imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit_transform(X[:, 1:])

labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

X = X[:, 1:]
# rotulos binários para nao interferir nos calculos
D = pd.get_dummies(X[:, 0])
X = np.insert(X, 0, D.values, axis=1)

xTrain, xtest, yTrain, yTest = train_test_split(X, Y, test_size=0.2)
