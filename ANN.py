import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)

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

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 8)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
"""
# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(16, activation = 'relu', input_dim = 13))

# Adding the second hidden layer
model.add(Dense(units = 7, activation = 'relu'))

# Adding the third hidden layer
#model.add(Dense(units = 6, activation = 'relu'))

# Adding the output layer

model.add(Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = model.predict(X_test)

# Calculating Root Mean Squared Error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))

# Visualizing Values 
plt.scatter(y_test, y_pred, s = 2.5, c = 'black')
plt.plot([0, 20000], [0, 20000], linestyle = '--', c = 'gray')
plt.title('Prediction by ANN Method (RMS error = %0.0f)' % rmse)
plt.ylabel('Predicted Value')
plt.xlabel('Real Value')
