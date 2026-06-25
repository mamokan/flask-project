import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pickle
df = pd.read_csv('student-lifestyle-and-stress-dataset.csv')

df["Student_Type"] = df["Student_Type"].map({"school": 0, "college": 1,'working_student': 2})
df = df.dropna()

print(df)

#🤖 Model Training
X = df.drop("Stress_Level", axis=1)
y = df["Stress_Level"]
"""
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 70)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
classifier =RandomForestClassifier()
classifier.fit(x_train_scaled, y_train)
y_pred_log = classifier.predict(x_test_scaled)
with open('model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

print("Model successfully trained and saved as model.pkl")
"""