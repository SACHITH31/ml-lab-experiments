import numpy as np
import pandas as pd
from sklearn import svm #Used for classification and regression

iris = pd.read_csv(r"C:\Users\SACHITH\Downloads\Iris - Iris.csv")
iris.head()

X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X =scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
