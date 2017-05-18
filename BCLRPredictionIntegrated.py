#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
LogisticRegression Prediction of Benign/Malignant Breast Cancer
=========================================================
This example uses all features of `Breast-Cancer` dataset from URL:https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/,
We initialize a binary classifier to classify benign / malignant breast cancer tumors with LogisticRegression Model,
At the same time, the performance of the model was evaluated by the more detailed evaluation index.

"""
print(__doc__)

# Author: HangJie

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
# Data processing
# Create feature list
column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size',
                'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
                'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

# Read specified data from Internet
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)

# Replace '?' with standard missing value
data = data.replace(to_replace='?',value=np.nan)

# Discard data with missing values
data = data.dropna(how='any')
print('(size,dim):',data.shape)

# Data Standardization
# Randomly sample 25% of the data for testing, and the remaining 75% for training sets
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

# Standardized data to ensure that the variance of the characteristic data for each dimension is 1 and the mean is 0, so that the prediction results are not dominated by some characteristic values that are too large
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Model Training
# LogisticRegression and SGDClassifier Initialization
lr = LogisticRegression()
sgdc = SGDClassifier()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)
print("LR accuracy:",lr.score(X_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))


sgdc.fit(X_train,y_train)
sgdc_y_predict = sgdc.predict(X_test)
print("SGD accuracy:",sgdc.score(X_test,y_test))
print(classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant']))