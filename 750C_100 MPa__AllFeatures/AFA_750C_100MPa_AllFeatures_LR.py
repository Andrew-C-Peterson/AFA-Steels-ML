#I'm now looking at just the 750C, 100 MPa data
#Here I do a linear regression

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
from sklearn.linear_model import LinearRegression
import AFA_750C_100MPa_AllFeatures_Correlation as rank

#Import my data and scale
df = pd.read_csv("Data_AllFeatures_750C_100MPa.csv")
scaler = MinMaxScaler()

x = df.drop(['id','NAME','LMP','RT','T2','Stress'],axis=1)
x_scaled=scaler.fit_transform(x)
y = df['RT']
y_2 = df['LMP']

ranked = rank.ranking
n = [10,20,50,100,200, len(ranked)-1]
labels_ranked = []
z = ranked.index

for j in range(0, len(n)):
    N = n[j]
    labels_ranked.append([])
    for i in range(1,N+1):
        labels_ranked[j].append(z[i])

acc_features = []
acc_score = []
acc_score_r2=[]
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state= 1)


for i in range(0, len(labels_ranked)):
    model_rkf_ranked = LinearRegression()
    x_scaled_ranked=scaler.fit_transform(x[labels_ranked[i]])
    results_rkf_ranked = cross_val_score(model_rkf_ranked, x_scaled_ranked, y, cv = rkf)
    acc_features.append(n[i])
    if results_rkf_ranked.mean() >= 0:
        acc_score.append(100*results_rkf_ranked.mean()**.5)
    else:
        acc_score.append(-100*((-1*results_rkf_ranked.mean())**.5))
    acc_score_r2.append(100*results_rkf_ranked.mean())

plt.plot(acc_features, acc_score,'o-', label = 'PCC')
plt.plot(acc_features, acc_score_r2, '*-', label = 'R2')
plt.xlabel('# of features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs # of features for LR (RT)')
plt.legend()
plt.ylim([-100,100])
plt.figure()

kfold = KFold(n_splits= 5, shuffle = True, random_state = 1)
y_hat_cv = cross_val_predict(model_rkf_ranked,x_scaled_ranked, y,cv=kfold)



plt.plot(y, y_hat_cv,'o')
plt.xlabel('Actual RT')
plt.ylabel('Predicted RT')
plt.title('Actual vs Predicted Values for LR (RT)')
plt.plot([0,6000],[0,6000])
plt.figure()


acc_features_2 = []
acc_score_2 = []
acc_score_r2_2=[]

for i in range(0, len(labels_ranked)):
    model_rkf_ranked_2 = LinearRegression()
    x_scaled_ranked_2=scaler.fit_transform(x[labels_ranked[i]])
    results_rkf_ranked_2 = cross_val_score(model_rkf_ranked_2, x_scaled_ranked_2, y_2, cv = rkf)
    acc_features_2.append(n[i])
    if results_rkf_ranked_2.mean() >= 0:
        acc_score_2.append(100*results_rkf_ranked_2.mean()**.5)
    else:
        acc_score_2.append(-100*((-1*results_rkf_ranked_2.mean())**.5))
    acc_score_r2_2.append(100*results_rkf_ranked_2.mean())

plt.plot(acc_features_2, acc_score_2,'o-', label = 'PCC')
plt.plot(acc_features_2, acc_score_r2_2, '*-', label = 'R2')
plt.xlabel('# of features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs # of features for LR (LMP)')
plt.legend()
plt.ylim([-100,100])
plt.figure()    

kfold = KFold(n_splits= 5, shuffle = True, random_state = 1)
y_hat_cv_2 = cross_val_predict(model_rkf_ranked_2,x_scaled_ranked_2, y_2,cv=kfold)

plt.figure()
plt.plot(y_2, y_hat_cv_2,'o')
plt.xlabel('Actual LMP')
plt.ylabel('Predicted LMP')
plt.title('Actual vs Predicted Values for LR (LMP)')
plt.plot([21000,25000],[21000,25000])
plt.figure()