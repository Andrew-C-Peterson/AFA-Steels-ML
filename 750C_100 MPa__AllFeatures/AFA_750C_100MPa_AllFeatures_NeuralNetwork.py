import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score

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

X_train, X_test, y_train, y_test = train_test_split(x_scaled_ranked, y, test_size=0.20)

input_layer = Input(shape=(x_scaled_ranked.shape[1],))
dense_layer_1 = Dense(100, activation='relu')(input_layer)
dense_layer_2 = Dense(50, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(25, activation='relu')(dense_layer_2)
output = Dense(1)(dense_layer_3)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

history = model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_split=0.2)

pred_train = model.predict(X_train)
pred = model.predict(X_test)

plt.plot(y_train,pred_train,'*')
plt.plot(y_test, pred,'^')
print(r2_score(y_test, pred))
print(r2_score(y_train, pred_train))