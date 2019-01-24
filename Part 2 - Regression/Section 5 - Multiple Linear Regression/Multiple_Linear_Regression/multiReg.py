# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:09:03 2019

@author: aztek
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset =pd.read_csv('50_Startups.csv')

print(dataset)

X =dataset.iloc[:,:-1].values
Y =dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=1/3,random_state =0)
