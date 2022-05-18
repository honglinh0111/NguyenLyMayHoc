# 1 #

#a
import numpy as np
import pandas as pd
import sklearn as sk

rt = pd.read_csv("winequality-white.csv", delimiter = ";")
rt.iloc[:,0:12] #in ra 5 dong dau va 5 dong cuoi
rt.iloc[1:5,0:11] #in ra dong 1->4, cot 0->10
rt.iloc[:,0:11] #in ra thuoc tinh cua tap du lieu winequality-white

#b
len(rt) #Co 4898 phan tu
rt.quality #nhan la quality
np.unique(rt.quality) #co 7 nhan (3, 4, 5, 6, 7, 8, 9)

#c
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rt.iloc[:,0:11], rt.quality, test_size = 2/8.0,random_state = 5 )

X_train[0:8] #Lay du lieu huan luyen tai cac dong         :2262 3893 3323 1675 4882 3549 698 1811
y_train[0:8] #Nhan cua du lieu huan luyen tai cac dong la :6    5    6    5    5    6    5   5
X_test[8:10] #Lay du lieu de kiem tra tai dong        :4355 2669
y_test[8:10] #Nhan cua du lieu kiem tra tai 2 dong la :6    5

#d: Xay dung cay quyet dinh su dung chi so GINI, khong dung thu vien sklearn
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 7, min_samples_leaf = 5)
clf.fit(X_train, y_train)

#du doan nhan
y_pred = clf.predict(X_test)
y_test
len(y_test) #du lieu kiem tra la 1225

#e cho toan bo du lieu
#do chinh xac tong the
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred, labels=[3,4,5,6,7,8,9])

#ket qua do chinh xac tong the
#array([[  0,   0,   2,   3,   0,   0,   0],
#       [  0,   3,  24,  16,   0,   1,   0],
#       [  1,   8, 214, 122,   7,   3,   0],
#       [  1,   5, 135, 361,  80,   3,   0],
#       [  0,   2,  12, 105,  83,   2,   0],
#       [  0,   0,   0,  11,  19,   2,   0],
#       [  0,   0,   0,   0,   0,   0,   0]], dtype=int64)

a = []
a = confusion_matrix(y_test, y_pred, labels=[3,4,5,6,7,8,9])

for i in range(0, 7):
    sum = 0
    for j in range(0, 7):
       sum = sum + a[i][j]
    print("KQ = ", a[i][i]/sum)
#ket qua:
#KQ dong 1 =  0.0
#KQ dong 2 =  0.06818181818181818
#KQ dong 3 =  0.6028169014084507
#KQ dong 4 =  0.6170940170940171
#KQ dong 5 =  0.4068627450980392
#KQ dong 6 =  0.0625

#do chinh xac tren tap test
from sklearn.metrics import accuracy_score
print ("Accuracy is ", accuracy_score(y_test, y_pred)*100)
#ket qua do chinh xac tren tap test = 54.122

#f cho 6 phan tu dau tien
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test[0:6], y_pred[0:6], labels=np.unique(y_test))

#ket qua do chinh xac tong the
#array([[0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0],
#       [0, 0, 1, 1, 0, 0],
#       [0, 0, 0, 1, 1, 0],
#       [0, 0, 0, 2, 0, 0],
#       [0, 0, 0, 0, 0, 0]], dtype=int64)




