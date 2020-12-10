import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#Import my data and scale
df = pd.read_csv("Data_AllFeatures_750C_100MPa.csv")
scaler = MinMaxScaler()
x = df.drop(['id','NAME','LMP','RT','T2','Stress'],axis=1)
x_scaled=scaler.fit_transform(x)
y= df['LMP']

model = RandomForestRegressor(n_estimators = 20, random_state = 1)
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

model_ab = AdaBoostRegressor(n_estimators = 10, random_state = 1)
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state= 1)
results = cross_val_score(model_ab, x_scaled_ranked, y, cv = rkf)


ab_new = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 5, min_samples_split = 2, min_samples_leaf = 2, max_features = 'sqrt', random_state = 1), 
                           n_estimators = 50, loss = 'linear', 
                           learning_rate = 0.01, random_state = 1)

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state= 1)
results_ab = cross_val_score(ab_new, x_scaled_ranked, y, cv = rkf)

print(results.mean())
print(results_ab.mean())

score_n = []
for n in [10, 25, 50, 100, 200, 300, 400, 500]:
    ab_new = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 5, min_samples_split = 2, min_samples_leaf = 2, max_features = 'sqrt', random_state = 1), 
                           n_estimators = n, loss = 'linear', 
                           learning_rate = 0.01, random_state = 1)
    results_n = cross_val_score(ab_new, x_scaled_ranked, y, cv = rkf)
    score_n.append(results_n.mean())
    
plt.plot([10, 25, 50, 100, 200, 300, 400, 500],score_n, '*-')


print('Best Accuracy is: ', max(score_n))
