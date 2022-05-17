# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:18:29 2022

@author: LeongKY
"""
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from predictor_modules import ClassificationModeller

#%% Load model
MODEL_LOAD_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.pkl')
SCALER_LOAD_PATH = os.path.join(os.getcwd(), 'saved_model', 'scaler.pkl')
ENCODER_LOAD_PATH = os.path.join(os.getcwd(), 'saved_model', 'encoder.pkl')
DATA_LOAD_PATH = os.path.join(os.getcwd(), 'dataset', 'heart_patient.csv')
heart_attack_dict = {1:'high risk', 0:'low risk'}
                               
#%% Load scaler and encoder
model = pickle.load(open(MODEL_LOAD_PATH, 'rb'))
scaler = pickle.load(open(SCALER_LOAD_PATH, 'rb'))
encoder = pickle.load(open(ENCODER_LOAD_PATH, 'rb'))

#%% Deploy model (testing)
# instantiate class file for reusing previously defined functions
mod = ClassificationModeller()

# load provided dataset with 10 sets converted into CSV from DATA_PATH
patient_info = pd.read_csv(DATA_LOAD_PATH)
patient_info = mod.check_dupe(patient_info)
print(patient_info.head(10))

# extract features and labels for prediction
X = patient_info[['thalachh', 'oldpeak', 'slp', 'caa', 'thall']]
X = scaler.transform(X)
y = np.expand_dims(patient_info['output'], -1)
y = encoder.transform(y)

mod.model_scoring(model, X, y)

# print accuracy to 2 decimal places
accuracy = '{:.2f}'.format(model.score(X, y)*100)
print('\nThis model has an accuracy of '+ str(accuracy)
      + ' percent on the unseen dataset.')

#%% Streamlit app implementation
'''
This streamlit app used the developed model to determine if a patient
is high or low risk of heart attack based on their medical information.
'''
# prepare form with defined limits/selection using slider and selectbox
with st.form('Heart Attack Risk Prediction Form'):
    st.write('Patient\'s info')
    thalachh = st.slider('Maximum heart rate achieved - thalachh',
                         min_value=0,
                         max_value=250,
                         step=1)
    oldpeak = st.slider('ST depression induced by exercise relative\
                              to rest - oldpeak',
                              min_value=0.0,
                              max_value=10.0,
                              step=0.1)
    slp = st.selectbox('Slope of peak exercise ST segment - slp\
                       (0. Downslope, 1. Flat, 2. Upslope)',
                          options=[0, 1, 2])
    caa = st.selectbox('Number of major vessels colored by flouroscopy - caa',
                       options = [0, 1, 2, 3])
    thall = st.selectbox('Thalium stress test result - thall\
                          (1. Fixed defect, 2. Normal, 3. Reversable defect)',
                          options=[1, 2, 3])
    
    submitted = st.form_submit_button('Submit information')
    st.write(submitted)
    
# perform prediction when form is submitted
    if submitted == True:
        pat_info = np.array([thalachh, oldpeak, slp, caa, thall])
        pat_info = np.expand_dims(pat_info,0)
        pat_info = scaler.transform(pat_info)
        prediction = model.predict(pat_info)
        st.write(np.argmax(prediction))
        st.write('This patient is at ' 
                 + heart_attack_dict[np.argmax(prediction)]
                 + ' of heart attack.')