'''
Notes on unbalanced dataset based on Porto Seguro’s Safe Driver Prediction
'''

# start by getting an idea on class counts/plotting
import numpy as np
import pandas as pd

df_train = pd.read_csv('../input/train.csv')

target_count = df_train.target.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)');


# don't use metrics like accuracy, instead something like Normalized Gini Coefficient

# Resampling
# undersampling the dominant class or oversampling smaller class
# Simple way: draw randomly or drop some rows

# Class count
count_class_0, count_class_1 = df_train.target.value_counts()

# get class specific dfs
df_class_0 = df_train[df_train['target'] == 0]
df_class_1 = df_train[df_train['target'] == 1]

# Random under-sampling
df_class_0_under = df_class_0.sample(count_class_1) # randomly sample
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0) # concatenate

# Random over-sampling
df_class_1_over = df_class_1.sample(count_class_0, replace=True) # sample with replacement
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)


# For more sophisticated techniques can use
import imblearn
# TomekLinks from imblearn or ClusterCentroids or SMOTE from the same package
# or from imblearn.combine import SMOTETomek


# splitting dataset
from sklearn.model_selection import StratifiedKFold
kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)
# stratified k fold keeps target distribution consistent in each fold
# which is useful for imbalanced dataset 

# can then split it or do whatever with it
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]