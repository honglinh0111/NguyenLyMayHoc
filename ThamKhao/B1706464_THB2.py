#Ho va ten: Nguyen Thi Hong Gam
#MSSV: B1706464
#MaHP: CT202_Nhom sang 5

#1a
import sklearn as sk
import numpy as np
import pandas as pd
vd = pd.read_csv("winequality-red.csv", delimiter = ";")
vd.iloc[:,0:12] #show 5 dong dau vaf 5 dong cuoi
vd.iloc[1:5,0:11] #dong 1->4, cot 0->10
vd.iloc[:,0:11] #thuoc tinh cua tap winequality-red 
#1b
len(vd) #1599 phan tu
vd.quality #nhan la quality.
np.unique(vd.quality) #co 6 nhan
#1c
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vd.iloc[:,0:11], vd.quality, test_size = 2/8.0, random_state = 5)

X_train[0:8] #lay du lieu de huan luyen tai cac dong 376, 1048, 1477, 939, 358, 1588, 1457, 207
y_train[0:8] #nhan cua du lieu huan luyen tai cac dong la 6, 6, 7, 5, 7, 6, 5, 5
X_test[8:10] #lay du lieu de kiem tra tai dong 1414, 994
y_test[8:10] #nhan cua du lieu kiem tra tai cac dong la 5, 5

#1d
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 7, min_samples_leaf = 5)
clf.fit(X_train, y_train)

#du doan nhan
y_pred = clf.predict(X_test)
y_test
len(y_test) #du lieu kiem tra la 400
#1e
#do chinh xac tong the
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred, labels=[3,4,5,6,7,8])
#ket qua do chinh xac tong the
array([[  0,   1,   0,   1,   0,   0],
       [  1,   1,   6,   2,   1,   0],
       [  1,   3, 126,  48,   1,   0],
       [  1,   2,  42,  86,  15,   2],
       [  0,   0,   4,  36,  13,   0],
       [  0,   0,   0,   4,   3,   0]], dtype=int64)
	   
a = []
a = confusion_matrix(y_test, y_pred, labels=[3,4,5,6,7,8])

for i in range(0, 6):
	sum = 0
    for j in range(0, 6):
		sum = sum + a[i][j]
	print("KQ = ", a[i][i]/sum)

#ket qua: dong 1: 0
#dong 2: 0.0909
#dong 3: 0.70391
#dong 4: 0.58108
#dong 5: 0.24528
#dong 6: 0.0		
#do chinh xac tren tap test
from sklearn.metrics import accuracy_score
print ("Accuracy is ", accuracy_score(y_test, y_pred)*100)
#ket qua do chinh xac tren tap test = 56,5

#1f
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test[0:8], y_pred[0:8], labels=np.unique(y_test))
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 4, 0, 0, 0],
       [0, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0]], dtype=int64)
	   