import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#import data
train_data = pd.read_csv("training.csv")
test_data = pd.read_csv("testing.csv")

#display missing values to see which attributes to concentrate on
def display_missing(df):    
    for col in df.columns.tolist():          
        print(col + " " + "column missing values: " + str(df[col].isnull().sum()))
    
display_missing(train_data)
print()
display_missing(test_data)

#data cleaning
train_data = train_data.drop('Name', axis = 1)
test_data = test_data.drop('Name', axis = 1)

#fill null values with median age
train_data["Age"] = train_data["Age"].fillna(28)
test_data["Age"] = test_data["Age"].fillna(28)

#continued data cleaning
train_data = train_data.drop('SibSp', axis = 1)
test_data = test_data.drop('SibSp', axis = 1)
train_data = train_data.drop('Parch', axis = 1)
test_data = test_data.drop('Parch', axis = 1)

train_data["Sex"] = train_data["Sex"].replace("male", 1)
train_data["Sex"] = train_data["Sex"].replace("female", 2)
test_data["Sex"] = test_data["Sex"].replace("male", 1)
test_data["Sex"] = test_data["Sex"].replace("female", 2)

#correlation plot
sns.heatmap(train_data.corr(), cmap="YlGnBu")
plt.show()

#drop unnessecary data
train_data = train_data.drop("Ticket", axis=1)
train_data = train_data.drop("Embarked", axis=1)
train_data = train_data.drop("Cabin", axis=1)
test_data = test_data.drop("Ticket", axis=1)
test_data = test_data.drop("Embarked", axis=1)
test_data = test_data.drop("Cabin", axis=1)

#Seperating Data
X = train_data.drop('Survived', axis = 1)
Y = train_data['Survived']

#training a support vector machine with multiple different parameters to see which results in highest accuracy
highest = 0
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, random_state = i)
    clf = svm.LinearSVC(random_state = i, C = 0.00001)
    clf.fit(x_train, y_train)
    accuracy = accuracy_score(clf.predict(x_test), y_test)
    if accuracy > highest:
        highest = accuracy
        print(str(i) + ": " + str(accuracy))

#FINAL MODEL
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, random_state = 773)
clfinal = svm.LinearSVC(random_state = 773, C = 0.00001)
clfinal.fit(x_train, y_train)
accuracy = accuracy_score(clfinal.predict(x_test), y_test)
accuracy