# python-machine-learning-Predict-Insurance-Charge-premi-
How to predict charge of life insurance (premi) with simple linear regression
#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('insurance.csv')
dataset.head()

#define data feature and target

x= dataset.drop('charges', axis=1)
y = dataset['charges']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size = 0.25, random_state=0)

#fitting simple linear regression in to model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

#Predict test result

ypred = reg.predict(x_test)
ypred

