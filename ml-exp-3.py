# knn for classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

names = ['sepal-length', 'sepal_width', 'petal-length', 'petal-width', 'Class']

dataset=pd.read_csv(r"C:\Users\SACHITH\Downloads\Iris - Iris.csv",header=0,names=names)
dataset.head(10)

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(X_train)

X_train=scaler.transform(X_train)

X_test=scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))




# knn for regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

names = ['sepal-length', 'sepal_width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv(r"C:\Users\SACHITH\Downloads\Iris - Iris.csv", header=0, names=names)

# 🔥 Change here: we predict a numerical column instead of class
X = dataset.iloc[:, :-2].values   # take first 3 features
y = dataset.iloc[:, 3].values     # predict petal-width (continuous value)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 🔥 Change here: Regressor instead of Classifier
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# 🔥 Evaluation for regression
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

print("Predicted values:", y_pred)
print("Mean Squared Error:", mse)











