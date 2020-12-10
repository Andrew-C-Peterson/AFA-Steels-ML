#I'm going to start by taking a look at the AFA data

#I'd like to try a multivariable linear regression
#Then I'd like to try a multivariable polynomial regression
#Then I'd like to try a KNN regression

import pandas as pd

#Import my data
df = pd.read_csv("Data.csv")

#Describes all the variables and then prints the shape of the dataset
des = df.describe()
print(df.shape)


#Is there missing data? I only print the counts of missing data if it
#does not equal 166, which is the total length of my data set.
missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    x = missing_data[column].value_counts()
    if x[0] !=166:
        print(column)
        print (x)
        print("") 
        
df = df.drop(['id','NAME'],axis=1)

#I just want to get a quick look at the distribution of LMP
import matplotlib.pyplot as plt

plt.hist(df["LMP"])
# set x/y labels and plot title
plt.xlabel("LMP")
plt.ylabel("Counts")
plt.figure()

#We can look at the correlations
corr = df.corr()

import seaborn as sns
sns.regplot(x="Stress", y="LMP", data=df)
plt.figure()
sns.regplot(x="T2", y="LMP", data=df)
plt.figure()
sns.regplot(x="T2_VPV_M23C6", y="LMP", data=df)
plt.figure()
sns.regplot(x="T2_VPV_L12", y="LMP", data=df)

df['T2_VPV_Strength']= df['T2_VPV_L12']+df['T2_VPV_M23C6']+df['T2_VPV_NbC']+df['T2_VPV_M3B2_D5A']

plt.figure()
sns.regplot(x="T2_VPV_Strength", y="LMP", data=df)

corr['R2'] = corr['LMP']**2

corr.sort_values(by='R2', inplace=True, ascending = False)