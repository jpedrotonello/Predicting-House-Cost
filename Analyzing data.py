# Rent prediction using Random Forest Regression Method
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('houses_to_rent_v2.csv')

# Removing categorical data
dataset = dataset.drop(columns=['city', 'animal', 'furniture', 'hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)'])

# If floor = "-", we consider as if floor = 0
dataset['floor'] = dataset['floor'].replace('-',0)
dataset['floor'] = pd.to_numeric(dataset['floor']) # convert everything to float values

# Viewing graphs
dataset.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
plt.show()

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

dataset= remove_outlier(dataset, 'area')
dataset= remove_outlier(dataset, 'rooms')
dataset= remove_outlier(dataset, 'bathroom')
dataset= remove_outlier(dataset, 'parking spaces')
dataset= remove_outlier(dataset, 'floor')
dataset= remove_outlier(dataset, 'total (R$)')
dataset['area'].plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
dataset['rooms'].plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
dataset['bathroom'].plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
dataset['parking spaces'].plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
dataset['floor'].plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
dataset['total (R$)'].plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)

sns.pairplot(dataset,kind="reg")
