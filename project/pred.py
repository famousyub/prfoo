# -*- coding: utf-8 -*-
"""Copie de predictionprojet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19cmRuWvjOzGveLs6xMCaiFGNt3oN00Ja
"""

#visualiser les 5 premières lignes


#importation des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

#import data from csv
data=pd.read_csv("./content/test.csv",sep=";")

data.info()

data.columns

data1=data.copy()

"""---

### Analyse de la forme des données
"""

#ce sont des compteurs ,on peut les supprimer
data=data.drop(['ID_ministere','ID_projet'],axis=1)

df2=data.copy()

df2.info()

"""⛳ Il est super utile de distinguer quelles sont les variables quantitatives et quelles sont les qualitatives."""

df2.dtypes.value_counts()

df2.dtypes.value_counts().plot.pie()

"""

Vérifions s'il existe des valeurs manquantes,
sachant que dans notre dataset elles sont marquées par "?", on va donc les affecter la valeur NAN puis les afficher:"""

for i in df2.columns:
  df2[i]=df2[i].replace('?',np.NAN)

df2.head(5)

df2

df2

#Heatmap qui montre les valeurs manquantes
plt.figure(figsize=(10,5))
sns.heatmap(df2.isna(), cbar=True)

#Pourcentage des valeurs manquantes dans chaque colonne
((df2.isna().sum()/df2.shape[0])*100).sort_values(ascending=True)

df2.columns

df2.head(5)

# Supposons que votre dataframe s'appelle df2
df2['tauxRetard'] = df2['tauxRetard'].str.replace(',', '.').astype(float)

df2.dtypes.value_counts()

"""### Examen de la colonne target:

"""

#prepare variables for feature selection
y = data1['tauxRetard']
X = df2.loc[:, df2.columns != 'tauxRetard']

# X = X.apply(pd.to_numeric, errors='coerce')
# y = y.apply(pd.to_numeric, errors='coerce')

# Feature Importance:
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X,y)

plt.figure(figsize=(8,6))
ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(24).plot(kind='barh')
plt.show()

#Geting the 3 importante features
imp_features=list(ranked_features.nlargest(3).index)
print(imp_features)

X = df2.loc[:, df2.columns != 'tauxRetard']
y = df2['tauxRetard']

"""# Modelisation"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import r2_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

df2['tauxRetard']

y_train

model = GradientBoostingRegressor()

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle en utilisant la métrique d'erreur quadratique moyenne (MSE)
mse1 = mean_squared_error(y_test, y_pred)
r1 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse1)
print("Coefficient de détermination (R-squared)",r1)

# Tracer le diagramme de dispersion
plt.scatter(y_test, y_pred)

# Ajouter des labels et un titre
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Diagramme de dispersion des prédictions")

# Afficher le diagramme de dispersion
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialiser le modèle de Forêts aléatoires
model = RandomForestRegressor()

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)


# Évaluer les performances du modèle en utilisant la métrique d'erreur quadratique moyenne (MSE)
mse2 = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse2)
print("Coefficient de détermination (R-squared)",r2)

# Tracer le diagramme de dispersion
plt.scatter(y_test, y_pred)

# Ajouter des labels et un titre
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Diagramme de dispersion des prédictions")

# Afficher le diagramme de dispersion
plt.show()



from sklearn.tree import DecisionTreeRegressor

# Initialiser le modèle de l'arbre de décision
model = DecisionTreeRegressor()

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle en utilisant la métrique d'erreur quadratique moyenne (MSE)
mse3 = mean_squared_error(y_test, y_pred)
r3 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse3)
print("Coefficient de détermination (R-squared)",r3)

# Tracer le diagramme de dispersion
plt.scatter(y_test, y_pred)

# Ajouter des labels et un titre
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Diagramme de dispersion des prédictions")

# Afficher le diagramme de dispersion
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle en utilisant la métrique d'erreur quadratique moyenne (MSE)
mse4 = mean_squared_error(y_test, y_pred)
r4 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse4)
print("Coefficient de détermination (R-squared)",r4)

# Tracer le diagramme de dispersion
plt.scatter(y_test, y_pred)

# Ajouter des labels et un titre
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Diagramme de dispersion des prédictions")

# Afficher le diagramme de dispersion
plt.show()

# Préparer les caractéristiques pour une prédiction
nouvelles_caracteristiques = [[60, 45, -15]]

# Effectuer la prédiction
prediction = model.predict(nouvelles_caracteristiques)

# Afficher la prédiction
print("La prédiction du taux de retard est :", prediction)