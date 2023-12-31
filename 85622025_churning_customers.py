# -*- coding: utf-8 -*-
"""85622025_Churning_Customers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HW5Rb86_hbNknFvlqajcbySUYlDXeLOL
"""

!pip install --upgrade scikit-learn
#!pip install --upgrade keras
!pip install scikeras
import pandas as pd
import csv
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

inputdata = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Churning/CustomerChurn_dataset.csv')
inputdata.info()

import matplotlib.pyplot as plt
import seaborn as sns

for column in inputdata.columns:
    plt.figure(figsize= (2,2))
    sns.histplot(inputdata[column], bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()
    print()

inputdata

# import matplotlib.pyplot as plt
# import seaborn as sns
# for column in inputdata.columns:
#     plt.figure(figsize= (5,5))
#     sns.histplot(inputdata[column], bins=30, kde=True)
#     plt.title(f'Distribution of {column}')
#     plt.show()

num = inputdata.select_dtypes(['int64','float64']).columns
num_data = inputdata[num]
num_data

"""# Exploratory Data Aanalysis
Histograms and boxplots for numerical data
Cross tabulation for Categorical data
"""

import matplotlib.pyplot as plt
import seaborn as sns

num_data.hist(bins=30, figsize=(5, 5))
plt.show()

num_data.boxplot(figsize=(5, 5))
plt.show()

cat = inputdata.select_dtypes(include =['object']).columns
cat_data = inputdata[cat]
cat_data

for feature in cat:
    plt.figure(figsize=(5, 5))
    sns.countplot(x=feature, hue='Churn', data=inputdata)
    plt.show()

for feature in inputdata.columns:
    plt.figure(figsize=(5, 5))
    sns.countplot(x=feature, hue='Churn', data=inputdata)
    plt.show()

cross_tabs = {}

for col in cat_data.columns:
    cross_tabs[col] = pd.crosstab(cat_data['Churn'], cat_data[col])

for col, cross_tab in cross_tabs.items():
    print(f"Cross-tabulation between churn and {col}:\n")
    print(cross_tab)
    print()

"""# Encoding categorical data and joining with numerical data"""

cat_data_fact = cat_data.apply(lambda x: pd.factorize(x)[0])
cat_data_fact

new_data = pd.concat([num_data, cat_data_fact], axis = 1)
new_data.info()

"""**Creating** dataframes to be used"""

y = new_data['Churn']
X = new_data.drop(['Churn','customerID'], axis=1)
for feature in X.columns:
    plt.figure(figsize= (3,3))
    sns.kdeplot(X[y == 0][feature], label='Not Churned', shade=True)
    sns.kdeplot(X[y == 1][feature], label='Churned', shade=True)
    plt.title(f'Distribution of {feature} by Churn')
    plt.legend()
    plt.show()
    print()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# Feature Selection using SelectKBest"""

from sklearn.feature_selection import SelectKBest, chi2, f_regression,f_classif
k_best = SelectKBest(score_func=f_classif, k=7)
X_train_scaled = k_best.fit_transform(X_train_scaled, y_train)
X_train_scaled

selected_feature_indices = k_best.get_support(indices=True)
X_train = X_train.iloc[:, selected_feature_indices]
X_test = X_test.iloc[:, selected_feature_indices]
X_train[:20]

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train

"""EXPLORING RELATIONSHIP BETWEEN EACH COLUMN AND CHURNING"""



"""# FUNCTIONAL MODEL CREATION"""

import tensorflow.keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model

import keras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

X_train_scaled, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def create_model(hidden_layer_size=32, learning_rate=0.001):
    inputs = Input(shape=(X_train_scaled.shape[1],))
    x = Dense(hidden_layer_size, activation="relu")(inputs)
    x = Dense(128, activation="relu")(x)
    z = Dense(64, activation="relu")(x)
    y = Dense(1, activation="sigmoid")(z)
    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create KerasClassifier with your model and hyperparameter options
model_wrapper = KerasClassifier(model=create_model, epochs=30, batch_size=32, verbose=0,hidden_layer_size=32,learning_rate=0.001)

"""# Hyperparameter tuning using GridSearch"""

param_grid = {
    'hidden_layer_size': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1]
}
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
grid_search = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=3)
grid_result = grid_search.fit(X_train_scaled, y_train)
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Accuracy: {grid_result.best_score_}")

best_params = grid_result.best_params_

best_model = create_model(hidden_layer_size=best_params['hidden_layer_size'], learning_rate=best_params['learning_rate'])

best_model.summary()

"""MODEL EVALUATION"""

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
best_model.fit(X_train_scaled, y_train, epochs=30, validation_data=(X_val, y_val), callbacks=[early_stopping])
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

from sklearn.metrics import accuracy_score, roc_auc_score
predictions = best_model.predict(X_test)
predictions_binary = (predictions > 0.5).astype(int)
print('actual')
print(predictions)
# print("Predictions:")
# print(predictions_binary)

print(predictions[0][0]*100)

accuracy = accuracy_score(y_test, predictions_binary)
auc_score = roc_auc_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
print(f'AUC Score: {auc_score}')

from joblib import dump, load
with open('/content/drive/My Drive/Colab Notebooks/Churning/churnmodel.joblib','wb') as f:
  dump(best_model,f)