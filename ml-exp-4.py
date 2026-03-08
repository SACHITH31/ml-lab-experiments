import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree']
df=pd.read_csv(r"C:\Users\SACHITH\Downloads\diabetes - diabetes (1).csv")
df.head(5)
# Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
# 0	6	148	72	35	0	33.6	0.627	50	1
# 1	1	85	66	29	0	26.6	0.351	31	0
# 2	8	183	64	0	0	23.3	0.672	32	1
# 3	1	89	66	23	94	28.1	0.167	21	0
# 4	0	137	40	35	168	43.1	2.288	33	1

print(df.columns) # Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'], dtype='str')
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'] 
X = df[feature_cols] 
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # Accuracy: 0.6796536796536796

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # Accuracy: 0.7705627705627706
