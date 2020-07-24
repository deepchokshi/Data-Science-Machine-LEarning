# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:09:12 2020

@author: Deep Chokshi
"""


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sn
from sklearn.linear_model import LinearRegression

#Importing CSV file and store into dataframe.
dataframe = pd.read_csv("House_Price.csv", header=0)

#Looking at top 5 row of datasets
print (dataframe.head())

#Looking at shape of data (no.of rows and no. of coloums)
print (dataframe.shape)

#Looking Extended Data Dictionary
print (dataframe.describe())

#Looking at information of data (counts and datatype)
print (dataframe.info())

#Ploting scatter plot to visulize (prise to number of rooms)
_ = sns.jointplot(x='n_hot_rooms', y='price', data=dataframe)
_ = sns.jointplot(x='rainfall', y='price', data=dataframe)

#Plotting categorical data
_ = sns.countplot(x='airport', data=dataframe)
_ = sns.countplot(x='waterbody', data=dataframe)
_ = sns.countplot(x='bus_ter', data=dataframe)
''' Observation till now
    1. Missing value in n_hot_beds
    2. Skewness or outliers in crimerate
    3. Outliers in n_hot_rooms and rainfall
    4. Bus_ter has only "Yes" Values'''
    
#Removing outlier using capping & flooring method for n_hot_rooms.
upper_limit = np.percentile(dataframe.n_hot_rooms,[99])[0]                        #Find the 99th percentile value
print (dataframe.n_hot_rooms[(dataframe.n_hot_rooms > upper_limit)])              #before removing outlier

#Change the value which is more than the 3 time of the upper limit anf assing it with 3*upper_limit.
dataframe.n_hot_rooms[(dataframe.n_hot_rooms > 3*upper_limit)] = 3*upper_limit
print (dataframe.n_hot_rooms[(dataframe.n_hot_rooms > upper_limit)])              #After removing outlier

#Removing outlier using capping & flooring method for rainfall.
lower_limit = np.percentile(dataframe.rainfall,[1])[0]                            #Find the 1st percentile calue
print (dataframe.rainfall[(dataframe.rainfall < lower_limit)])                    #befor removing outlier

#Change the value which is less than the 0.3 time of the lower limit anf assing it with 0.3*upper_limit
dataframe.rainfall[(dataframe.rainfall < 0.3*lower_limit)] = 0.3*lower_limit
print (dataframe.rainfall[(dataframe.rainfall < lower_limit)])                    #after removing outlier

#Filling the missing value in n_hot_beds.
print (dataframe.info())                                                          #check in which column data is missing
dataframe.n_hos_beds = dataframe.n_hos_beds.fillna(dataframe.n_hos_beds.mean())
print (dataframe.info())                                                          #Again check is it filled or not

#Variable Trasnformation
_ = sns.jointplot(x="crime_rate", y="price", data=dataframe)                      #Look at scatter plot its look like log curv
dataframe.crime_rate = np.log(1+dataframe.crime_rate)
_ = sns.jointplot(x="crime_rate", y="price", data=dataframe)                      #Look again at scatter plot, relationship looks more linear.

#As we notice there are 4 employment center dist1, dist2, dist3, dist4 in our data, let take average of it and make new column
dataframe['avg_dist'] = (dataframe.dist1+dataframe.dist2+dataframe.dist3+
                         dataframe.dist4)/4
del dataframe['dist1']                                                            #Deleting 4 employment column because we have taken its average
del dataframe['dist2']  
del dataframe['dist3']  
del dataframe['dist4']
del dataframe['bus_ter']                                                          #because it has on 1 type of value.
dataframe.describe()                                                              #Look at dataframe now
dataframe.info()

#Dummy Variable
dataframe = pd.get_dummies(dataframe)
dataframe.head() #Look at dataset now airpot_No is not needed and waterbody_None aslo
del dataframe['airport_NO']
del dataframe['waterbody_None']
dataframe.head()

#Correlation matrix
dataframe.corr()                                                                 #Take a observation we can see that park and air_qual is hoghly corelated so it make cause multi co-linearality
del dataframe['parks']
dataframe.head()
#Data Preprocessing Completed

#Simple Linear Regression using statsmodel
X = sn.add_constant(dataframe["room_num"])
lm = sn.OLS(dataframe["price"],X).fit()
lm.summary()

#Linear Regession usinf sklearn
y = dataframe["price"]
X = dataframe[["room_num"]]                                                      #x should be a 2D array so we use[[]]
lm2 = LinearRegression()                                                         #Creating Linear Regration object
lm2.fit(X,y)
print (lm2.intercept_, lm2.coef_)

help(lm2)                                                                        #to get help regarding object

lm2.predict(X)                                                                   #predicting the values of y on model which we generated
sns.jointplot(x= dataframe['room_num'], y= dataframe['price'], data= dataframe, kind='reg')

