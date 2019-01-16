# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 23:43:41 2019

@author: aztek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset =pd.read_csv('Salary_Data.csv')

print(dataset)

X =dataset.iloc[:,:-1].values
Y =dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=1/3,random_state =0)

# =============================================================================
# fitting the leanear regression model
# =============================================================================


from sklearn.linear_model import LinearRegression
lr =LinearRegression()
lr.fit(X_train,Y_train)

# =============================================================================
# preddicting the data for the test cases
# =============================================================================

y_pred =lr.predict(X_test)

# =============================================================================
# Plotting the data 
# =============================================================================

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,lr.predict(X_train),color='blue')
plt.title("Salary vs Experience")
plt.xlabel("yrs of Experience")
plt.ylabel("Sallary")
plt.show()



