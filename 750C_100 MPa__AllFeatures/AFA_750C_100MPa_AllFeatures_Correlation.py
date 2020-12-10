#I'm now looking at just the 750C, 100 MPa data
#Here, I just want to look at correlation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import neighbors
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold


#Import my data and scale
df = pd.read_csv("Data_AllFeatures_750C_100MPa.csv")
scaler = MinMaxScaler()

plt.plot(df['RT'], df['LMP'],'*')
plt.xlabel('RT')
plt.ylabel('LMP')
plt.figure()

x = df.drop(['id','NAME','LMP','RT'],axis=1)
x_scaled=scaler.fit_transform(x)
y = df['RT']
y_2 = df['LMP']

corr = df.drop(['id','NAME','T2','Stress','RT'],axis=1).corr()
ranking = (corr['LMP']**2)**.5
ranking.sort_values(inplace=True, ascending = False)

ax1 = sns.distplot(y, hist=False, color="r", label="RT")

ax1 = sns.distplot(y_2, hist=False, color="b", label="LMP")

plt.figure()


