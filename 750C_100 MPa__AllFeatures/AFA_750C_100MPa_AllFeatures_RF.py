#I'm now looking at just the 750C, 100 MPa data
#Here I do RF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
import AFA_750C_100MPa_AllFeatures_Correlation as rank
from sklearn.ensemble import RandomForestRegressor

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
    model_rkf_ranked = RandomForestRegressor(n_estimators = 10, random_state = 1)
    x_scaled_ranked=scaler.fit_transform(x[labels_ranked[i]])
    results_rkf_ranked = cross_val_score(model_rkf_ranked, x_scaled_ranked, y, cv = rkf)
    acc_features.append(n[i])
    if results_rkf_ranked.mean() >= 0:
        acc_score.append(100*results_rkf_ranked.mean()**.5)
    else:
        acc_score.append(-100*((-1*results_rkf_ranked.mean())**.5))
    acc_score_r2.append(100*results_rkf_ranked.mean())



kfold = KFold(n_splits= 5, shuffle = True, random_state = 1)
y_hat_cv = cross_val_predict(model_rkf_ranked,x_scaled_ranked, y,cv=kfold)

acc_features_2 = []
acc_score_2 = []
acc_score_r2_2=[]

for i in range(0, len(labels_ranked)):
    model_rkf_ranked_2 = RandomForestRegressor(n_estimators = 10, random_state = 1)
    x_scaled_ranked_2=scaler.fit_transform(x[labels_ranked[i]])
    results_rkf_ranked_2 = cross_val_score(model_rkf_ranked_2, x_scaled_ranked_2, y_2, cv = rkf)
    acc_features_2.append(n[i])
    if results_rkf_ranked_2.mean() >= 0:
        acc_score_2.append(100*results_rkf_ranked_2.mean()**.5)
    else:
        acc_score_2.append(-100*((-1*results_rkf_ranked_2.mean())**.5))
    acc_score_r2_2.append(100*results_rkf_ranked_2.mean())

plt.plot(acc_features_2, acc_score_2,'o-', label = 'PCC, LMP')
#plt.plot(acc_features_2, acc_score_r2_2, '*-', label = 'R2, LMP')
plt.plot(acc_features, acc_score,'o-', label = 'PCC, RT')
#plt.plot(acc_features, acc_score_r2, '*-', label = 'R2, RT')
plt.xlabel('# of features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs # of features for RF')
plt.legend()
plt.ylim([-100,100])
plt.figure()    

kfold = KFold(n_splits= 5, shuffle = True, random_state = 1)
x_scaled_ranked_2=scaler.fit_transform(x[labels_ranked[0]])
y_hat_cv_2 = cross_val_predict(model_rkf_ranked_2,x_scaled_ranked_2, y_2,cv=kfold)

plt.figure()
plt.plot(y_2, y_hat_cv_2,'o', label = 'LMP')
plt.plot(y, y_hat_cv,'o', label = 'RT')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values for RF, 10 Features')
plt.legend()
plt.plot([0,25000],[0,25000])
plt.figure()

model = RandomForestRegressor(n_estimators = 10, random_state = 1)

model.fit(x_scaled, y_2)

importances = model.feature_importances_
 
indices = np.argsort(importances)[::-1] 

z_1 = x.columns

RF_ranking =[]
for i in indices:
    RF_ranking.append(z_1[i])

labels_ranked_2 = []

for j in range(0, len(n)):
    N = n[j]
    labels_ranked_2.append([])
    for i in range(0,N):
        labels_ranked_2[j].append(RF_ranking[i])
        
        
#I want to do this for this ranking method now, instead of the PCC. It could
#be better
    
acc_features_3 = []
acc_score_3 = []
acc_score_r2_3=[]

for i in range(0, len(labels_ranked)):
    model_rkf_ranked_3 = RandomForestRegressor(n_estimators = 10, random_state = 1)
    x_scaled_ranked_3=scaler.fit_transform(x[labels_ranked_2[i]])
    results_rkf_ranked_3 = cross_val_score(model_rkf_ranked_3, x_scaled_ranked_3, y_2, cv = rkf)
    acc_features_3.append(n[i])
    if results_rkf_ranked_3.mean() >= 0:
        acc_score_3.append(100*results_rkf_ranked_3.mean()**.5)
    else:
        acc_score_3.append(-100*((-1*results_rkf_ranked_3.mean())**.5))
    acc_score_r2_3.append(100*results_rkf_ranked_3.mean())

plt.plot(acc_features_3, acc_score_3,'o-', label = 'PCC, RF ranking')
plt.plot(acc_features_2, acc_score_2,'o-', label = 'PCC, PCC ranking')
plt.plot(acc_features_3, acc_score_r2_3, '*-', label = 'R2, RF ranking')
plt.plot(acc_features_2, acc_score_r2_2, '*-', label = 'R2, PCC ranking')
plt.xlabel('# of features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs # of features for RF (LMP)')
plt.legend()
plt.ylim([-100,100])
plt.figure()    





