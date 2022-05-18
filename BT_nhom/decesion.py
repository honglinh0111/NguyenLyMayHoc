import numpy as np
import pandas as pd
import sklearn as sk

data = pd.read_csv('csv_result-messidor_features.csv')

#print(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:19], data.Class, test_size = 1/8.0,random_state = 5 )

X_train[0:8] 
y_train[0:8] 
X_test[8:10] 
y_test[8:10]

#Xay dung mo hinh cay quyet dinh du tren entropy
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
clf.fit(X_train, y_train)

print (clf.predict(X_test))
