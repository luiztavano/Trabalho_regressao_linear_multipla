import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

import numpy as np

#Ler e formatar a base de dados
data = pd.read_csv('advertising.csv')
data = data.drop(data.columns[0], axis = 1)



### STATSMODELS ###
# Criar o modelo e encaixar as variáveis
lm1 = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

# print the coefficients
lm1.params

# exibir resumo com as informações
lm1.summary()

### SCIKIT-LEARN ###
#Criação das variáveis x e y
feature_cols = ['TV', 'Radio', 'Newspaper']
x = data[feature_cols]
y = data.Sales

# Criar o modelo de previsão e encaixar as variáveis
lm2 = LinearRegression()
lm2.fit(x, y)

# print the coefficients
print (lm2.intercept_)
print (lm2.coef_)

#REALIZAR PREVISÃO 1

#Valores
TV = 1000
Radio = 1000
Newspaper = 1000

valores = np.array([TV,Radio,Newspaper])
valores = valores.reshape(1,-1)

#Resultado da previsão
lm2.predict(valores)

#REALIZAR PREVISÃO 2

#Valores
TV = 1500
Radio = 1500
Newspaper = 0

valores = np.array([TV,Radio,Newspaper])
valores = valores.reshape(1,-1)

#Resultado da previsão
lm2.predict(valores)

#REALIZAR PREVISÃO 3

'#Valores
Tv = data["TV"].mean()
Radio = data["Radio"].mean()
Newspaper = data["Newspaper"].mean()

valores = np.array([Tv,Radio,Newspaper])
valores = valores.reshape(1,-1)

#Resultado da previsão
lm2.predict(valores)'

#Coeficientes
# Intercept    2.938889
# TV           0.045765
# Radio        0.188530
# Newspaper   -0.001037

