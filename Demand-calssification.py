#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 05:34:37 2020

@author: meshal
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats

train = pd.read_csv('train.csv') # Users who were 60 days enrolled, churn in the next 30
test = pd.read_csv('test.csv') 

#### EDA ####


train.head(5) # Viewing the Data
train.columns
train.describe() # Distribution of Numerical Variables

# Removing NaN

dataset = train.drop(columns = ['long', 'lat','id'])
dataset.shape
dataset.isna().sum()
dataset.dropna(inplace=True)
dataset.isnull().sum().max() #just checking that there's no missing

#descriptive statistics summary
dataset['quantity'].describe()
dataset['price'].describe()

#histogram
sn.distplot(dataset['quantity'])
sn.distplot(dataset['price'])

#scatter plot grlivarea/saleprice
var = 'price'
data = pd.concat([dataset['quantity'], dataset[var]], axis=1)
data.plot.scatter(x=var, y='quantity', ylim=(0,800000))

##Relationship with categorical features

#box plot city/quantity
var = 'city'
data = pd.concat([dataset['quantity'], dataset[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
plt.suptitle('Box Plot city/quantity', fontsize=20)
fig = sn.boxplot(x=var, y="quantity", data=data)
fig.axis(ymin=0, ymax=800000);

#Time destrpution

var = 'date'
data = pd.concat([dataset['quantity'], dataset[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
plt.suptitle('Time destrpution for quantity', fontsize=20)
fig = sn.boxplot(x=var, y="quantity", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90)


#correlation matrix
corrmat = dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
plt.suptitle('Correlation matrix', fontsize=20)
sn.heatmap(corrmat, vmax=.8, square=True)



dataset3=dataset.copy()

## Correlation with Response Variable
dataset3.drop(columns = ['date',
                         'quantity',]
    ).corrwith(dataset.quantity).plot.bar(figsize=(20,10),
              title = 'Correlation with Response variable',
              fontsize = 15, rot = 45,
              grid = True)



#Out liars

#standardizing data
quantity_scaled = StandardScaler().fit_transform(dataset['quantity'][:,np.newaxis]);
low_range = quantity_scaled[quantity_scaled[:,0].argsort()][:10]
high_range= quantity_scaled[quantity_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


#standardizing data
price_scaled = StandardScaler().fit_transform(dataset['price'][:,np.newaxis]);
low_range = price_scaled[price_scaled[:,0].argsort()][:10]
high_range= price_scaled[price_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#The price maby have 1 outliaer we will deal with it later


#Low range values are similar and not too far from 0.
#High range values are far from 0 and the 4.something values are really out of range.


#Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.

#applying log transformation
dataset['quantity'] = np.log(dataset['quantity'])

#transformed histogram and normal probability plot
sn.distplot(dataset['quantity'], fit=norm);
fig = plt.figure()
res = stats.probplot(dataset['quantity'], plot=plt)



dataset['price'] = np.log(dataset['price'])
#check the price
sn.distplot(dataset['price'], fit=norm);
fig = plt.figure()
res = stats.probplot(dataset['price'], plot=plt)

dataset2 = dataset.copy()
#convert categorical variable into dummy
dataset2 = pd.get_dummies(dataset2.drop(columns='date',axis =1 ))
dataset2.shape

dataset4 = dataset.copy()












































