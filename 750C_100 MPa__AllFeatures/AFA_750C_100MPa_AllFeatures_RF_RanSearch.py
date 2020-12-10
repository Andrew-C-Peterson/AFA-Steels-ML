#Random search for hyperparameters, which will start improving the accuracy


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

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


#The following code prints the current parameters
#pprint(model_rf.get_params())

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1110, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 21)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 5, verbose=2, random_state=1, n_jobs = -1)

rf_random.fit(x_scaled_ranked,y)
best_parameters = rf_random.best_params_

best_model = rf_random.best_estimator_
results_random = cross_val_score(best_model, x_scaled_ranked, y, cv = rkf)

print(results.mean())
print(results_random.mean())
improve = results_random.mean()-results.mean()
print('Improvement of: ', improve)