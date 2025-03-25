from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# Exemplo de carregamento de dados
chp = pd.read_csv('housing.csv')
chp['latitude_longitude'] = chp['latitude'] * chp['longitude']

chp2 = pd.get_dummies(chp, columns=['ocean_proximity'], drop_first=True)
chp2 = chp2[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
             'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity_INLAND',
             'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
             'ocean_proximity_NEAR OCEAN']]
# Suponha que temos 'preco_aluguel' como variável alvo e o restante como features
X = chp2.drop('median_house_value', axis=1)
y = chp2['median_house_value']

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Criando o modelo
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=15,
    num_leaves=50,
    min_gain_to_split=0.001,  # Reduz o ganho mínimo para criar uma divisão
    min_child_samples=20,
    random_state=42
)

# Treinando
model.fit(X_train, y_train, eval_set=[
          (X_test, y_test)])

# Previsões
y_pred = model.predict(X_test)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')
