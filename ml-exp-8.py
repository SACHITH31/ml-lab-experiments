import pandas as pd #Dataset handling
import numpy as np #Dataset handling
import matplotlib.pyplot as plt # Numerical operations
import seaborn as sns #Plotting graphs
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression

df_sal = pd.read_csv(r"C:\Users\SACHITH\Downloads\Salary_Data - Salary_Data - Salary_Data - Salary_Data.csv")
df_sal.head()

plt.scatter(df_sal['YearsExperience'], df_sal['Salary'], color = 'lightcoral')
#Plots individual data points.
#lightcoral sets point color.
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(True) #Removes border around the plot for cleaner appearance
plt.show()

X = df_sal.iloc[:, :1]; 
y = df_sal.iloc[:, 1:];

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

clf = LinearRegression()
clf.fit(X_train, y_train)

y_pred_test = clf.predict(X_test);
y_pred_train = clf.predict(X_train);

plt.scatter(X_train, y_train, color = 'lightcoral')
plt.plot(X_train, y_pred_train, color = 'firebrick')

plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Sal/Exp', loc='best', facecolor='white')
plt.box(False)
plt.show()

plt.scatter(X_test, y_test, color = 'lightcoral')
plt.plot(X_train, y_pred_train,color = 'firebrick')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Sal/Exp', loc='best',facecolor='white')
plt.box(False)
plt.show()

print(f'Coefficient: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')
