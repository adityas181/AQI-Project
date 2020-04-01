# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:49:38 2020

@author: aditya
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Data/Real-Data/Real_Combine.csv')

df.head()

sns.heatmap(df.isnull(),yticklabels=True,cbar=True,cmap='viridis')

df=df.dropna()
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X.isnull()
y.isnull()

sns.pairplot(df)

df.corr()

#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

corrmat.index

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)

X.head()

print(model.feature_importances_)


sns.distplot(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)


from sklearn.neighbors import KNeighborsRegressor

regressor=KNeighborsRegressor(n_neighbors=1)
regressor.fit(X_train,y_train)


print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))


print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))


from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)

score.mean()

prediction=regressor.predict(X_test)
sns.distplot(y_test-prediction)


plt.scatter(y_test,prediction)


accuracy_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsRegressor(n_neighbors=i)
    score=cross_val_score(knn,X,y,cv=10,scoring="neg_mean_squared_error")
    accuracy_rate.append(score.mean())

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
#plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
 #        markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsRegressor(n_neighbors=1)

knn.fit(X_train,y_train)
predictions = knn.predict(X_test)

sns.distplot(y_test-predictions)

plt.scatter(y_test,predictions)


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(X_train,y_train)
predictions = knn.predict(X_test)

sns.distplot(y_test-predictions)

plt.scatter(y_test,predictions)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

import pickle
# open a file, where you ant to store the data
file = open('knn_regression.pkl', 'wb')

# dump information to that file
pickle.dump(knn, file)