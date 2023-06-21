import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_text

data = pd.read_csv('../input/decision-trees-2022/train.csv')
print(data.head())

feature_cols = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
X = data.iloc[:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
Y = data.iloc[:,[11]].values

clf = tree.DecisionTreeClassifier(max_depth = 5)
clf = clf.fit(X, Y)

tree_rules = export_text(clf, feature_names = feature_cols)

print(tree_rules)

predictions = clf.predict(listedData)
predictions = predictions.tolist()
