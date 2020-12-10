#A grid search for hyperparameter tuning
#After narrowing down the best parameters using random search
#The grid search will narrow it down more specifically

from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor

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

model_rf = RandomForestRegressor(n_estimators = 10, random_state = 1)
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state= 1)
results = cross_val_score(model_rf, x_scaled_ranked, y, cv = rkf)



model_random = RandomForestRegressor(n_estimators = 343, min_samples_split = 5, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 20, bootstrap= False, random_state = 1)
results_random = cross_val_score(model_random, x_scaled_ranked, y, cv = rkf)

score = results.mean()
score_random = results_random.mean()

param_grid = {
    'max_depth': [10, 20],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [3, 5, 7]}

rf = RandomForestRegressor(n_estimators = 343, max_features= 'sqrt', bootstrap= False, random_state = 1)
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(x_scaled_ranked, y)

print(grid_search.best_params_)

best_grid = grid_search.best_estimator_

results_grid = cross_val_score(best_grid, x_scaled_ranked, y, cv = rkf)

score_grid = results_grid.mean()

print(score)
print(score_random)
print(score_grid)



mss =  grid_search.best_params_['min_samples_split']
msl =  grid_search.best_params_['min_samples_leaf']
md =  grid_search.best_params_['max_depth']

score_n = []
rkf_2 = RepeatedKFold(n_splits=5, n_repeats=1, random_state= 1)
for n in [10, 25, 50, 100, 200, 300, 400, 500]:
    rf = RandomForestRegressor(n_estimators = n, max_features= 'sqrt', bootstrap= False, random_state = 1, min_samples_split = mss, min_samples_leaf = msl, max_depth = md)
    results_n = cross_val_score(rf, x_scaled_ranked, y, cv = rkf_2)
    score_n.append(results_n.mean())

plt.plot([10, 25, 50, 100, 200, 300, 400, 500],score_n, '*-')


print('Best Accuracy is: ', max(score_n))

