#I'm going to start by taking a look at the AFA data

#I'd like to try a multivariable linear regression
#Then I'd like to try a multivariable polynomial regression
#Then I'd like to try a KNN regression

#Linear Regression
#Going to use k-fold cross validation with k =5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

#Import my data
df = pd.read_csv("Data.csv")
scaler = MinMaxScaler()
df[['T2','Stress','dT']]=scaler.fit_transform(df[["T2","Stress","dT"]])

labels = ['T2', 'Stress','dT','T2_VPV_FCC','T2_VPV_FECR_B2','T2_VPV_L12','T2_VPV_LAVES_C14','T2_VPV_M23C6','T2_VPV_M2B_CB','T2_VPV_M3B2_D5A','T2_VPV_NbC','T2_VPV_NIAL_B2','T2_VPV_SIGMA']

x = df[labels]
y = df[['LMP']]

#First, let's split the data into a test set and train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#Create the linear regression
lm = LinearRegression()

#And now perform the regression
lm.fit(x_train,y_train)
y_hat_train=lm.predict(x_train)
y_hat_test = lm.predict(x_test)

print("R2 for train data: ", lm.score(x_train,y_train))
print("R2 for test data: ", lm.score(x_test,y_test))

#Distribution plot
plt.figure(figsize=(12, 10))
ax1 = sns.distplot(y_train, hist=False, color="r", label="Actual Value")
sns.distplot(y_hat_train, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for LMP for Train')
plt.xlabel('LMP')
plt.ylabel('Proportion of Alloys')
plt.show()
plt.close()
plt.figure()

ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_hat_test, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for LMP for Test')
plt.xlabel('LMP')
plt.ylabel('Proportion of Alloys')
plt.show()
plt.close()
plt.figure()


#This doesn't appear to be overfitting the data, becuase it still works reasonably 
#Well for the test data. That's good, so now let's do the cross-val

Rcross = cross_val_score(lm, x, y, cv=5)

print("R for cross val is: ", Rcross.mean())

y_hat_cv = cross_val_predict(lm,x, y,cv=5)

plt.figure(figsize=(10, 12))
plt.plot(y, y_hat_cv,'o')
plt.plot([20000,26000],[20000,26000])
plt.xlabel('Actual LMP')
plt.ylabel('Fitted LMP')
plt.title('Actual vs Fitted Values')
plt.figure()

y['y_hat'] = y_hat_cv

y_pcc = y.corr()

ax1 = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(y_hat_cv, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for LMP for CV')
plt.xlabel('LMP')
plt.ylabel('Proportion of Alloys')
plt.show()
plt.close()
plt.figure()

y_pcc = np.array(y_pcc)
print('Accuracy is: ', y_pcc[1,0])
