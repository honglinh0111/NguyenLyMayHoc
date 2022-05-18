import numpy as np
import pandas as pd
import sklearn as sk

#Doc du lieu
gt = pd.read_csv('BT2.csv', index_col=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(gt.iloc[:,0:3], gt.Nhan, test_size = 1/3.0,random_state = 5 )

X_train[0:2]
y_train[0:2]

#Xay dung mo hinh cay quyet dinh du tren entropy
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
clf.fit(X_train, y_train)

#Du doan nhan cua phan tu mới tới có thông tin chiều cao=135, độ dài mái tóc = 39 và giọng nói có giá trị là 1 
print (clf.predict([[135,39,1]]))

