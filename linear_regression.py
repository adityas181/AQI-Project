# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 01:44:59 2020

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

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,y_train)

regressor.coef_

regressor.intercept_

print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))


print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))



from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)


score.mean()

coeff_df = pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])
coeff_df

prediction=regressor.predict(X_test)


sns.distplot(y_test-prediction)


plt.scatter(y_test,prediction)


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

import pickle
# open a file, where you ant to store the data
file = open('regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(regressor, file)