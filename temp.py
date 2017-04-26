
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 21:41:00 2017

@author: Shubha
............. Random Forest implementation using Sklearn............
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn import metrics

#output_file = "out.csv"

dataset = pd.read_csv('new_dataset.csv') 
#testdata = pd.read_csv('test.csv') 

y = np.array(dataset)[:,20]
#le = preprocessing.LabelEncoder()

train = np.array(dataset)[:,0:20]
#test = np.array(datase)[:,0:]

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size = 0.2)
#x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.3)

rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)

print(metrics.accuracy_score(y_test, pred))
print pred


