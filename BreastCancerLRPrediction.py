#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
LogisticRegression Prediction of Benign/Malignant Breast Cancer
=========================================================
This example uses the feature 'Clump Thickness'and 'Cell Size' of the `Breast-Cancer` dataset,
We initialize a binary classifier to classify benign / malignant breast cancer tumors with LogisticRegression Model.

"""
print(__doc__)

# Author: HangJie

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Get data from Datasets
df_train = pd.read_csv('E:/python/Datasets/Breast-Cancer/breast-cancer-test.csv')
df_test = pd.read_csv('E:/python/Datasets/Breast-Cancer/breast-cancer-test.csv')

# Select 'Clump Thickness' and 'Cell Size' as feature, and construct positive and negative samples of test set
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# LogisticRegression Initialize
lr = LogisticRegression()

# Use the top 10 training samples to learn the intercept and the coefficients of the line
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
accuracy = lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type'])
print('Testing accuracy(top 10 training samples):',accuracy)
intercept = lr.intercept_
coef = lr.coef_[0,:]

# intercept = np.random.random([1])
# coef = np.random.random([2])

lx = np.arange(0,12)
ly = (-intercept-lx * coef[0]) / coef[1]

# Draw a random line
plt.plot(lx,ly,c='green')

# Draw a benign tumor sample point marked red 'o'
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')

# Draw a malignant tumor sample point marked black 'x'
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')

# Draw x,y axis
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
