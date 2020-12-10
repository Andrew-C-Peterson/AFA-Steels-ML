#Random search for hyperparameters, which will start improving the accuracy


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, DecisionTreeRegressor
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

model_ab = AdaBoostRegressor(n_estimators = 10, random_state = 1)
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state= 1)
results = cross_val_score(model_ab, x_scaled_ranked, y, cv = rkf)


#The following code prints the current parameters
#pprint(model_rf.get_params())

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 510, num = 6)]
# Loss
loss = ['linear', 'square', 'exponential']
# Method of selecting samples for training each tree
l_rate = [0.1, 0.25, 0.5, 1, 2, 4, 7, 10]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'loss': loss,
               'learning_rate': l_rate}

ab = AdaBoostRegressor()
ab_random = RandomizedSearchCV(estimator = ab, param_distributions = random_grid, n_iter = 25, cv = 5, verbose=2, random_state=1, n_jobs = -1)

ab_random.fit(x_scaled_ranked,y)
best_parameters = ab_random.best_params_

best_model = ab_random.best_estimator_
results_random = cross_val_score(best_model, x_scaled_ranked, y, cv = rkf)

print(results.mean())
print(results_random.mean())
improve = results_random.mean()-results.mean()
print('Improvement of: ', improve)
print('Best parameters: ', best_parameters)


