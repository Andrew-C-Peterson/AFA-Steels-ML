#Comparison of a decision tree to a random forest to a boosting algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


#Import my data and scale
df = pd.read_csv("Data_AllFeatures_750C_100MPa.csv")
scaler = MinMaxScaler()
x = df.drop(['id','NAME','LMP','RT','T2','Stress'],axis=1)
x_scaled=scaler.fit_transform(x)
y= df['LMP']

model = RandomForestRegressor(n_estimators = 10, random_state = 1)
model.fit(x_scaled, y)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1] 
z = x.columns
RF_ranking =[]
for i in indices:
    RF_ranking.append(z[i])
    if len(RF_ranking) == 20:
        break

x_scaled_ranked = scaler.fit_transform(x[RF_ranking])
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state= 1)

model_tree = DecisionTreeRegressor(random_state = 1)
results_tree = cross_val_score(model_tree, x_scaled_ranked, y, cv = rkf)

model_rf = RandomForestRegressor(n_estimators = 10, random_state = 1)
results_rf = cross_val_score(model_rf, x_scaled_ranked, y, cv = rkf)

model_ab = AdaBoostRegressor(n_estimators = 10, random_state = 1)
results_ab = cross_val_score(model_ab, x_scaled_ranked, y, cv = rkf)

model_gb = GradientBoostingRegressor(n_estimators = 10, random_state = 1) 
results_gb = cross_val_score(model_gb, x_scaled_ranked, y, cv = rkf)

print(results_tree.mean())
print(results_rf.mean())
print(results_ab.mean())
print(results_gb.mean())






