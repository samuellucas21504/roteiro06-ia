from sklearn import preprocessing
import pandas as pd
import sklearn.metrics as metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
import numpy as np

print('''
1. Crie um dataset simples para classificação de dados e comente os resultados obtidos com a técnica de
KNN.

 Código abaixo. Sobre os resultados, com apenas 1 n_neighbor já foi possível receber a predição
 padrão, isso pois, as características inseridas são muito simples. Caso fossem mais dados isso,
 provavelmente, mudaria.
''')

# Criar um dataset

# Primeira Característica

treinou_ontem = ['Não', 'Sim', 'Sim', 'Sim', 'Não', 'Não', 'Não', 'Não', 'Sim']

# Segunda Característica

dormiu_bem = ['Sim', 'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim']

# Alvo
correr = ['Sim', 'Sim', 'Não', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não']

# criando labelEncoder
le = preprocessing.LabelEncoder()
# Convertendo string labels para números.
treinou_encoded = le.fit_transform(treinou_ontem)
dormiu_encoded = le.fit_transform(dormiu_bem)
print(treinou_encoded)
print(dormiu_encoded)
# convertendo string labels para números
alvo = le.fit_transform(correr)
print(alvo)
# Combinando clima e temp em um única lista de tuplas
carac = list(zip(treinou_encoded, dormiu_encoded))

modelo = KNeighborsClassifier(n_neighbors=1)
# Treinando o modelo usando os ajustes de treinamento.
modelo.fit(carac, alvo)

# Predito
predito = modelo.predict([[0, 1]])  # 0: Não Treinou, 2: Dormiu Bem
print(predito)

print('''
2 - Aplique os passos descritos na atividade 2 no dataset do Titanic, e comente os resultados obtidos.
(https://www.kaggle.com/datasets/jamesleslie/titanic-cleaned-data)
''')

# Ler os csv's com o pandas
train = pd.read_csv('./input/train_clean.csv')
test = pd.read_csv('./input/test_clean.csv')

# Remover a coluna Cabin, já que ela é nula e não usaremos
train = train.drop('Cabin', axis=1)
test = test.drop('Cabin', axis=1)

# Preencher as colunas nulas com 0, nesse caso apenas a coluna Survived é nula então
# será colocado que ninguem sobreviverá
test = test.fillna(0)

# Cria x e y de treinamento
x_train = pd.get_dummies(train[["Pclass", "Sex", "SibSp", "Parch"]])
y_train = train["Survived"]

# Cria x e y de teste
x_test = pd.get_dummies(test[["Pclass", "Sex", "SibSp", "Parch"]])
y_test = test["Survived"]

# treina o modelo utilizando o KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# prediz
predicted = model.predict(x_test)

# compara a taxa de acerto com o csv de teste
acc = metrics.accuracy_score(y_test, predicted)

print("Taxa de acerto do predito com o real: ", acc)

print('''
    3. Utilize a ideia da atividade 3 e descubra a média com desvio padrão e o histograma de acurácia para
1000 treinamentos dos algoritmos de KNN aplicado no dataset do câncer de mama.
''')

# dataset
cancer_mama = load_breast_cancer()

# caracteristicas e alvo
X = cancer_mama.data
y = cancer_mama.target

# normalizando
scaler = Normalizer()
scaler.fit(X)
X = scaler.transform(X)
scores = []
for i in range(1000):
 X_train, X_test, y_train, y_test = train_test_split(X,y)
 model = KNeighborsClassifier()
 model.fit(X_train,y_train)
 precisao = model.score(X_test,y_test)
 scores.append(precisao)

print("Média: {:.2f}%".format(np.mean(scores)*100))
print("Desvio padrão: {:.2f}%".format(np.std(scores)*100))
import matplotlib.pyplot as plt
import seaborn as sns
sns.displot(scores)
plt.yticks([])
plt.title("Acurácias do k-NN")
plt.show()
