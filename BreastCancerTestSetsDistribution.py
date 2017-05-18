#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Distribution of Benign/Malignant Breast Cancer Test Sample
=========================================================
This example uses the feature 'Clump Thickness'and 'Cell Size' of the `Breast-Cancer` dataset, in
order to observe the distribution of breast cancer test sample in two-dimensional feature space.

"""
print(__doc__)

# Author: HangJie


import pandas as pd
import matplotlib.pyplot as plt

# Get data from Datasets
df_train = pd.read_csv('E:/python/Datasets/Breast-Cancer/breast-cancer-test.csv')
df_test = pd.read_csv('E:/python/Datasets/Breast-Cancer/breast-cancer-test.csv')

# Select 'Clump Thickness' and 'Cell Size' as feature, and construct positive and negative samples of test set
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# Draw a benign tumor sample point marked red 'o'
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')

# Draw a malignant tumor sample point marked black 'x'
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')

# Draw x,y axis
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
