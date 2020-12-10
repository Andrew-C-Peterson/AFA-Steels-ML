#I'm going to start by taking a look at the AFA data

#I'd like to try a multivariable linear regression
#Then I'd like to try a multivariable polynomial regression
#Then I'd like to try a KNN regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Import my data
df = pd.read_csv("Data.csv")

labels = ['T2', 'Stress','dT','T2_VPV_FCC','T2_VPV_FECR_B2','T2_VPV_L12','T2_VPV_LAVES_C14','T2_VPV_M23C6','T2_VPV_M2B_CB','T2_VPV_M3B2_D5A','T2_VPV_NbC','T2_VPV_NIAL_B2','T2_VPV_SIGMA']

x = df[labels]
y = df[['LMP']]

#Create the linear regression
lm = LinearRegression()

#And now perform the regression
lm.fit(x,y)
y_hat=lm.predict(x)

#Data visualization
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(y, y_hat)
plt.xlabel('Actual LMP')
plt.ylabel('Fitted LMP')
plt.title('Actual vs Fitted Values for Linear')
plt.figure()

#Polynomial order 2, all features
pr=PolynomialFeatures(degree=2)
x_pr=pr.fit_transform(x)
poly=LinearRegression().fit(x_pr,y)
y_hat_pr=poly.predict(x_pr)

plt.figure(figsize=(width, height))
sns.regplot(y, y_hat_pr)
plt.xlabel('Actual LMP')
plt.ylabel('Fitted LMP')
plt.title('Actual vs Fitted Values for Poly (2)')
plt.figure()


K=4
KNN = neighbors.KNeighborsRegressor(n_neighbors = K)
KNN.fit(x, y)
y_hat_knn=KNN.predict(x)

   
plt.figure(figsize=(width, height))
sns.regplot(y, y_hat_knn)
plt.xlabel('Actual LMP')
plt.ylabel('Fitted LMP')
plt.title('Actual vs Fitted Values for KNN')
plt.figure()

#Distribution plot
plt.figure(figsize=(12, 10))
ax1 = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(y_hat_pr, hist=False, color="b", label="Fitted Values Poly(2)" , ax=ax1)
sns.distplot(y_hat, hist=False, color="g", label="Fitted Values Linear" , ax=ax1)
sns.distplot(y_hat_knn, hist=False, color="y", label="Fitted Values KNN" , ax=ax1)
plt.title('Actual vs Fitted Values for LMP')
plt.xlabel('LMP')
plt.ylabel('Proportion of Alloys')
plt.show()
plt.close()
plt.figure()


print('Linear fit has MSE:', mean_squared_error(y, y_hat))
print('Poly(2) fit has MSE: ', mean_squared_error(y,y_hat_pr))
print('KNN(4) fit has MSE: ', mean_squared_error(y,y_hat_knn))

y['y_hat'] = y_hat
y['y_hat_pr']=y_hat_pr
y['y_hat_knn']=y_hat_knn

y_pcc = y.corr()

print(y_pcc)


#These all look decent for fitting the data, but I now need to 
#use test/train split to see how well this actually works on predicting
#I will have a seperate file for each, and then I will compare