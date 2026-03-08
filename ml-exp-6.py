import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

df_net = pd.read_csv(r"C:\Users\SACHITH\Downloads\Social_Network_Ads - Social_Network_Ads.csv")

df_net.drop(columns = ['User ID'], inplace=True)

le = LabelEncoder()

df_net['Gender']= le.fit_transform(df_net['Gender'])

X = df_net.iloc[:, :-1].values
y = df_net.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')
print(f"F1 Score : {f1_score(y_test, y_pred)}")
cf_matrix = confusion_matrix(y_test, y_pred)
cf_matrix
print(classifier.predict(sc.transform([[1, 45, 97000]])))
