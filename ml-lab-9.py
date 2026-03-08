import pandas as pd

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree','age','lable']
pima=pd.read_csv(r"C:\Users\SACHITH\Downloads\diabetes - diabetes.csv")
pima.head(5)

# Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
# 0	6	148	72	35	0	33.6	0.627	50	1
# 1	1	85	66	29	0	26.6	0.351	31	0
# 2	8	183	64	0	0	23.3	0.672	32	1
# 3	1	89	66	23	94	28.1	0.167	21	0
# 4	0	137	40	35	168	43.1	2.288	33	1

feature_cols = ['pregnant', 'insulin', 'bmi','age','glucose','bp','pedigree']
print(pima.columns) # Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'], dtype='str')

feature_cols = ['Pregnancies', 'Insulin', 'BMI','Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
X = pima[feature_cols]
y = pima['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=16, max_iter=500)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix) 
# [[118  12]
 # [ 26  36]]
