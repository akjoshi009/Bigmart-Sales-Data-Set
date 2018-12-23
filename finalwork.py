# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 23:54:46 2018

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import xgboost as xgb



data=pd.read_csv("D://24projects//Project 3//Train_UWu5bXk.csv")
datatest=pd.read_csv("D://24projects//Project 3//Test_u94Q5KV.csv")
datat=pd.read_csv("D://24projects//Project 3//Test_u94Q5KV.csv")

data.head()
data.dtypes
data.isnull().sum()
data.Item_Weight.value_counts()


data.Item_Weight.mean()
data.Item_Weight.median()
data.Item_Weight.fillna(data.Item_Weight.median(),inplace=True)

data.Outlet_Size.value_counts()
data.Outlet_Size.hist(bins=10)
data.boxplot(column="Item_Outlet_Sales",by="Outlet_Size")
data.boxplot(column="Outlet_Establishment_Year",by="Outlet_Size")
gender = {'Medium': 3,'Small': 2,'High':1} 
data.Outlet_Size = [gender[item] for item in data.Outlet_Size] 
#data.boxplot(column="Outlet_Location_Type",by="Outlet_Size")
data.Outlet_Size.fillna("Medium",inplace=True)


#updating values

data.Item_Fat_Content.value_counts()
data.boxplot(column="Item_Outlet_Sales",by="Item_Fat_Content")
gender = {'Low Fat': 5,'Regular': 4,'LF':3,'low fat':2,'reg':1} 
data.Item_Fat_Content = [gender[item] for item in data.Item_Fat_Content] 


data.Outlet_Location_Type.value_counts()
gender = {'Tier 3': 3,'Tier 2': 2,'Tier 1':1} 
data.Outlet_Location_Type = [gender[item] for item in data.Outlet_Location_Type] 


data.Outlet_Type.value_counts()
gender = {'Supermarket Type1': 4,'Grocery Store': 3,'Supermarket Type3':2,'Supermarket Type2':1} 
data.Outlet_Type = [gender[item] for item in data.Outlet_Type] 




data = data.drop('Item_Type', 1)
data=data.drop('Outlet_Identifier',1)
data=data.drop('Item_Fat_Content',1)
data=data.drop('Item_Weight',1)
data.shape

data[data.columns[1:8]]

#find depedent Variables

X2 = sm.add_constant(data[data.columns[1:6]])
est = sm.OLS(data['Item_Outlet_Sales'], X2)
est2 = est.fit()
print(est2.summary())

reg = LinearRegression().fit(data[data.columns[1:6]],data["Item_Outlet_Sales"])
reg.score(data[data.columns[1:6]],data["Item_Outlet_Sales"])


#Genrate Test data in appropriate format



datatest.isnull().sum()
datatest = datatest.drop('Item_Type', 1)
datatest=datatest.drop('Outlet_Identifier',1)
datatest=datatest.drop('Item_Fat_Content',1)
datatest=datatest.drop('Item_Weight',1)
 
datatest.Item_Weight.median()
datatest.Item_Weight.fillna(datatest.Item_Weight.median(),inplace=True)

datatest.Outlet_Size.value_counts()
datatest.Outlet_Size.fillna("Medium",inplace=True)
gender = {'Medium': 3,'Small': 2,'High':1} 
datatest.Outlet_Size = [gender[item] for item in datatest.Outlet_Size] 

datatest.Item_Fat_Content.value_counts()
gender = {'Low Fat': 5,'Regular': 4,'LF':3,'low fat':2,'reg':1} 
datatest.Item_Fat_Content = [gender[item] for item in datatest.Item_Fat_Content] 


datatest.Outlet_Location_Type.value_counts()
gender = {'Tier 3': 3,'Tier 2': 2,'Tier 1':1} 
datatest.Outlet_Location_Type = [gender[item] for item in datatest.Outlet_Location_Type] 


datatest.Outlet_Type.value_counts()
gender = {'Supermarket Type1': 4,'Grocery Store': 3,'Supermarket Type3':2,'Supermarket Type2':1} 
datatest.Outlet_Type = [gender[item] for item in datatest.Outlet_Type] 

datatest.head()


#usig Randome forest Regresser
regr = RandomForestRegressor(max_depth=2, random_state=0,
                             n_estimators=500)
regr.fit(data[data.columns[1:6]],data["Item_Outlet_Sales"])
datat.dtypes
pred=regr.predict(datatest[datatest.columns[1:6]])



#using Xgboost
xgb = xgb.XGBRegressor(n_estimators=50, learning_rate=0.09, gamma=0, subsample=0.85,
                           colsample_bytree=1, max_depth=7)
xgb.fit(data[data.columns[1:6]],data["Item_Outlet_Sales"])
pred=xgb.predict(datatest[datatest.columns[1:6]])

datat["Item_Outlet_Sales"]=pred
newdf=datat[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
newdf.to_csv("D://24projects//Project 3//output.csv", encoding='utf-8', index=False)

datat["Item_Outlet_Sales"].isnull().sum()












