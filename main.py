import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Cargar datos de entrenamiento y prueba
data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

#Elijo ùnicamente las columnas que me interesan para el modelo
X_test = test_data[['Pclass', 'Sex', 'Age', 'Fare']]
categories = data[['Pclass', 'Sex', 'Age', 'Fare']]
result = data[['Survived']] 

#Limpiar datos
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}) #Convertir sexo a 0 o 1
data['Age'].fillna(data['Age'].median(), inplace=True)  #Rellenar edades faltantes
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

#Modelo secuencial, quise hacer un Perceptron multicapa con 4 inputs.
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=4))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Entrenar el modelo, puse 1000 épocas que puede ser exagerado. Pero no debería tomar más de 1 minuto en entrenar
model.fit(categories, result, epochs=1000, batch_size=64)

#Preparar los datos de prueba y hacer predicciones
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  #Convertir predicciones a 0 o 1

#Las predicciones o sea 0 si murió y 1 si sobrevivió se guardan en esa columna
test_data['Survived'] = predictions

#calcular la precisión del modelo
accuracy = model.evaluate(categories, result)

#Innecesario pero es para ver en consola la precisión, es estimada y se guarda en el segundo valor del array.
#OJO AL PIOJO!! La precisión que me devuelve el evaluate, es cercana a la de Kaggle, pero siempre es menos precisa.
print(f'Precisión: {accuracy}')

#Guardar resultados en un archivo CSV con el nombre de la precisión, esto sse puede subir de una a kaggle
test_data[['PassengerId', 'Survived']].to_csv(f'titanic/predictions_{accuracy[1]}.csv', index=False)
