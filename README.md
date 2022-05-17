![badge](http://ForTheBadge.com/images/badges/made-with-python.svg)

# heart_attack_predictor
 This program is used to develop a model which assess the risk of a patient for heart attack based on the patient's medical data using RandomTreeClassifer.

# How to use
Clone the repository and use the following scripts per your use case:
1. train.py is the script that is used to train the model
2. app.py is the script that is used to deploy the model, validated with heart_patient.csv which contains 10 rows of patient data. It also includes the   streamlit app implementation which provides a GUI to predict the heart attack risk of a patient during deployment.
3. predictor_modules.py is the class file that contains the defined functions used during training and evaluation for added robustness and reusability of the processes used.
4. The saved model, scaler, and encoder are available in .pkl format in the 'saved_model' folder.
5. The original dataset (heart.csv) and validation dataset (heart_patient.csv) are available in the 'dataset' folder.
6. Screenshots of the streamlit app implementation and train/test/validation results are available in the 'results' folder.

# Results
The model developed using Random Tree Classfier and optimized using GridSearchCV was scored using accuracy and f1-score, attaining 80% accuracy on the test dataset and 9/10 correct predictions on the validation dataset.

Streamlit app implementation GUI:

![App](https://github.com/khaiyuann/heart_attack_predictor/blob/main/results/streamlit_app_implementation.png)

Train/test results (achieved 80% accuracy and f1 score):
![Results1](https://github.com/khaiyuann/heart_attack_predictor/blob/main/results/train_test_results.png)

Validation results (achieved 90% accuracy and f1 score):
![Results2](https://github.com/khaiyuann/heart_attack_predictor/blob/main/results/validation_result.png)

# Credits
Big thanks to Rashik Rahman (Kaggle: rashikrahmanpritom) for providing the Heart Attack Classification & Prediction Dataset used for the training of the model on Kaggle. 
Check it out here for detailed information: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
