# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:33:32 2022

This class file contains the functions for performing heart attack prediction
on patients based on their medical data.

@author: LeongKY
"""
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

class ClassificationModeller():
    def __init__(self):
        pass
    
    def check_dupe(self, data):
        '''
        This function is used to check if the dataset contains duplicate
        entries, and removes the duplicate entries, if any.

        Parameters
        ----------
        data : DataFrame
            Dataset to inspect for duplicates.

        Returns
        -------
        data : DataFrame
            Dataset with duplicate entries removed.

        '''
        if data.duplicated().sum() > 0:
            dupes = int(data.duplicated().sum())
            data.drop_duplicates()
            print('Removed ' + str(dupes) + ' duplicates in the dataset.')
        else:
            print('No duplicates found in this dataset')
        return data
    
    
    def check_correlation(self, data):
        '''
        This function is used to check the correlation between the features and
        label, and plots the corresponding heatmap for interpretation.

        Parameters
        ----------
        data : DataFrame
            Dataframe containing the data to check correlation for.

        Returns
        -------
        None.

        '''
        correlation = data.corr()
        sns.heatmap(abs(correlation), annot=True, cmap=plt.cm.Reds)
        plt.show()
        
    def scale_data(self, scaler, data, path):
        '''
        This function is used to scale data and export the scale parameters
        based on the selected scaler object.

        Parameters
        ----------
        scaler : Scaler object
            Instantiated scaler object of choice.
        data : array
            Data to scale.
        path : path
            Directory to save the scaler parameters in .pkl format for
            deployment.

        Returns
        -------
        scaled : array
            Scaled data.

        '''
        scaled = scaler.fit_transform(data)
        pickle.dump(scaler, open(path, 'wb'))
        return scaled
        
    def encode_data(self, encoder, data, path):
        '''
        This function is used to encode data and export the encoding parameters
        based on the selected encoder object.

        Parameters
        ----------
        encoder : Encoder object
            Instantiated encoder object of choice.
        data : array
            Data to encode.
        path : path
            Directory to save the encoder parameters in .pkl format for
            deployment.

        Returns
        -------
        data : array
            Encoded data.

        '''
        data = encoder.fit_transform(np.expand_dims(data,-1))
        pickle.dump(encoder, open(path, 'wb'))
        return data
    
    def grid_search(self, classifier, params, folds=10, scoring='f1_weighted'):
        '''
        This function is used to perform grid-search on the parameter range to
        optimize the selected classfier hyperparameters.

        Parameters
        ----------
        classifier : Classifier object
            Instantiated classifier object of choice.
        params : dict
            Dictionary containing the parameter and corresponding ranges to
            search.
        folds : int, optional
            Number of cross validation folds to search. The default is 10.
        scoring : str, optional
            Type of scoring to evaluate model. The default is 'f1_weighted'.

        Returns
        -------
        model : model
            Selected model from grid search with the best parameters.

        '''
        model = GridSearchCV(estimator=classifier,
                             param_grid=params,
                             scoring=scoring,
                             cv=folds,
                             n_jobs=-1)
        return model
    
    def model_scoring(self, model, X_test, y_test):
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print('\nClassification Report:\n')
        print(classification_report(y_true, y_pred))
        print('Confusion Matrix:\n')
        print(confusion_matrix(y_true, y_pred))
    
    def model_export(self, model, path):
        '''
        This function is used to export a machine learning model to .pkl 
        format for deployment.

        Parameters
        ----------
        model : model
            Machine learning model to be exported.
        path : path
            Directory to save the model in .pkl format for deployment.

        Returns
        -------
        None.

        '''
        with open(path, 'wb') as file:
            pickle.dump(model, file)