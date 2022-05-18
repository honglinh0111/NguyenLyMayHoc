#lay file iris truc tiep tu sklearn
from sklearn.datasets import load_iris
iris_dt = load_iris()
iris_dt.data[1:5] #thuoc tinh cua lop iris
iris_dt.target[1:5] #gia tri cua nhan/class

#phan chia tap du lieu
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dt.data, iris_dt.target, test_size = 1/3.0, random_state = 5)

X_train[1:6]
X_train[1:6, 1:3]
y_train[1:6]
X_test[6:10]
y_test[6:10]

#xay dung mo hinh cay quyet dinh
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 3, min_samples_leaf = 5)
clf_gini.fit(X_train, y_train)

#du doan nhan/class
y_pred = clf_gini.predict(X_test)
y_test
clf_gini.predict([[4, 4, 3, 3]])

#tinh do chinh xac cho gia tri du doan
from sklearn.metrics import accuracy_score
print ("Accuracy is ", accuracy_score(y_test, y_pred)*100)

#B mot so cach doc dl dau vao
#doc bang thu vien pandas
import numpy as np
import pandas as pd
dt5 = pd.read_csv("iris_data.csv")
dt5[1:5]
len(dt5)
dt5.petalLength[1:5]

#tao bien luu tru dl
#C bai toan hoi quy
#doc dl va bien dl tu pandas
import pandas as pd
dulieu = pd.read_csv("housing_RT.csv", index_col=0)
dulieu.iloc[1:5,]

#su dung nghi thuc Hold-out
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dulieu.iloc[:,1:5], dulieu.iloc[:,0], test_size = 1/3.0, random_state = 100)
X_train[1:5]
X_test[1:5]

#xay dung mo hinh
from sklearn.tree import DecisionTreeRegressor
rgs = DecisionTreeRegressor(random_state = 0)
rgs.fit(X_train, y_train)

#du bao vaf danh gia mo hinh
y_pred = rgs.predict(X_test)
y_test[1:5]
y_pred[1:5]

#danh gia ket qua du doan gia tri nha thong qua chi so MSE va RMSE
import numpy as np #de su dung sqrt
from sklearn.metrics import mean_squared_error
err = mean_squared_error(y_test, y_pred)
err #686543211.95116
np.sqrt(err) #can bac 2 cua err = 26201.96961...

