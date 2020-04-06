# Rent prediction using Random Forest Regression Method
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('houses_to_rent_v2.csv')
X = dataset.iloc[:, [True, True, True, True, True, True, True, True, True, False, True, False, False]].values
y = dataset.iloc[:, -4].values

for i in range(0, len(X)):
    if X[i][5] == '-':
        X[i][5] = 0

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 6] = labelencoder_X_1.fit_transform(X[:, 6])
labelencoder_X_2 = LabelEncoder()
X[:, 7] = labelencoder_X_2.fit_transform(X[:, 7])
labelencoder_X_3 = LabelEncoder()
X[:, 0] = labelencoder_X_2.fit_transform(X[:, 0])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.002, random_state = 5)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
"""
# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 8)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
"""

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor

# Obtaining the best number of estimators
best_rmse = 1e10
for i in range(1, 100, 4):
    # Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(n_estimators = i, random_state = 0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print('Loading ' + str(i) + '% RMSE = ' + str(rmse))
    if rmse < best_rmse:
        best_rmse = rmse
        best_y_pred = y_pred
        
plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(best_y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction by Random Forest Regression Method (RMS error = %0.0f)' % best_rmse)
plt.legend()
plt.show()