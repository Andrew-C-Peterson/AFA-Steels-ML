


from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

#Import my data and scale
df = pd.read_csv("Data_750C_100MPa.csv")
scaler = MinMaxScaler()

labels = ['dT','T2_VPV_FCC','T2_VPV_FECR_B2','T2_VPV_L12','T2_VPV_LAVES_C14','T2_VPV_M23C6','T2_VPV_M2B_CB','T2_VPV_M3B2_D5A','T2_VPV_NbC','T2_VPV_NIAL_B2','T2_VPV_SIGMA']

x = df[labels]
x_scaled=scaler.fit_transform(x[labels])
y = df['LMP']

labels_corr = ['LMP']+labels
corr = df[labels_corr].corr()
ranking = (corr['LMP']**2)**.5
ranking.sort_values(inplace=True, ascending = False)

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state= 1)
model_rkf = RandomForestRegressor(n_estimators = 10, random_state = 1)
results_rkf = cross_val_score(model_rkf, x_scaled, y, cv = rkf, scoring = 'r2')
print("Acc for 10 iterations: R2 is %.2f%%, R is %.2f%%" % ((results_rkf.mean()*100.0), 100*results_rkf.mean()**.5))

kfold = KFold(n_splits= 5, shuffle = True, random_state = 1)
model_kfold = RandomForestRegressor(n_estimators = 10, random_state = 1)
y_hat_cv = cross_val_predict(model_kfold,x_scaled, y,cv=kfold)

plt.plot(y, y_hat_cv,'o')
plt.plot([21000,25000],[21000,25000])
plt.xlabel('Actual LMP')
plt.ylabel('Predicted LMP')
plt.title('Actual vs Predicted Values for RF, 11 Features')
plt.figure()

y_corr = y.to_frame()
y_corr.insert(1,'y hat', y_hat_cv)
y_pcc = np.array(y_corr.corr())

ax1 = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(y_hat_cv, hist=False, color="b", label="Predicted Values" , ax=ax1)
plt.title('Actual vs Predicted Values for RF, 13 Features')
plt.xlabel('LMP')
plt.ylabel('Proportion of Alloys')
plt.show()
plt.close()
plt.figure()

print('PCC accuracy for cross_val_predict is: %.2f%% ' % (y_pcc[1,0] * 100))

n = [4,8,len(labels)]
labels_ranked = []
z = ranking.index

for j in range(0, len(n)):
    N = n[j]
    labels_ranked.append([])
    for i in range(1,N+1):
        labels_ranked[j].append(z[i])
        
acc_features = []
acc_score = []
acc_score_r2 = []
for i in range(0,len(labels_ranked)):
    model_rkf_ranked = RandomForestRegressor(n_estimators = 10, random_state = 1)
    x_scaled_ranked=scaler.fit_transform(x[labels_ranked[i]])
    results_rkf_ranked = cross_val_score(model_rkf_ranked, x_scaled_ranked, y, cv = rkf, scoring = 'r2')
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
plt.title('Accuracy vs # of features for RF')
plt.legend()
plt.ylim([-50,100])
plt.figure()