# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:26:16 2022

@author: LeongKY
"""

#%% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from predictor_modules import ClassificationModeller

#%% Static paths
DATA_PATH = os.path.join(os.getcwd(), 'dataset', 'heart.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model','model.pkl')
SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'scaler.pkl')
ENC_PATH = os.path.join(os.getcwd(), 'saved_model', 'encoder.pkl')

#%% 1. Load data
df = pd.read_csv(DATA_PATH)

# instantiate class file
mod = ClassificationModeller()

#%% 2. Inspect data
print(df.describe().T)
print(df.info())

# boxplot to observe data distribution
df.plot(kind='box')
plt.show()

#%% 3. Clean data
# convert categorical columns to 'category' type
cat_col = ['exng', 'cp', 'fbs', 'restecg']
df[cat_col] = df[cat_col].astype('category')

# check for duplicated data and remove if present
df = mod.check_dupe(df)

# from data inspection, no NaN/suspicious data, proceed to select features
#%% 4. Select features
X = df.drop(labels='output', axis=1)
y = df['output']

# correlation heatmap for feature selection
correlation = mod.check_correlation(df)

# selected 5 features with highest correlation index
X = df[['thalachh', 'oldpeak', 'slp', 'caa', 'thall']]

#%% 5. Preprocess data
# scaling selected features
scaler = MinMaxScaler()
X_scaled = mod.scale_data(scaler, X, SCALER_PATH)

# encoding labels
encoder = OneHotEncoder(sparse=False)
y = mod.encode_data(encoder, y, ENC_PATH)

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.3,
                                                    random_state=17)

#%% 6. Create model
# selected Random Forest for classification
classifier = RandomForestClassifier()

# hyperparameter tuning using GridSearchCV to get best model in param range
params = {'max_depth':np.arange(2,10,1), 
          'n_estimators':np.arange(100, 500, 100)}

# train and evaluate model using weighted f1 scoring
grid_search = mod.grid_search(classifier, params, 10)
model = grid_search.fit(X_train, y_train)
print('This model has an f1 score of: '+ str(model.score(X_test, y_test)) 
      + ' with parameter ' + str(model.best_params_))

# model scoring
mod.model_scoring(model, X_test, y_test)

#%% 7. Export model
mod.model_export(model, MODEL_SAVE_PATH)