#BÀI THỰC HÀNH 3 - GIẢI THUẬT BAYES THƠ NGÂY 
#Họ Tên: Phạm Thị Hồng Linh
#MSSV: B1809365

import numpy as np
import pandas as pd
import random as rd

dt = pd.read_csv("winequality-white.csv", delimiter = ";")
print(dt)

#Câu 1
s=list(dt.iloc[:,:])
print(len(s))# co 12 thuoc tinh
print(np.unique(dt.quality))#Gia tri cua  nhan quality la (3 4 5 6 7 8 9)

#Câu 2
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
dt = shuffle(dt)
dt.reset_index (inplace = True, drop = True)
print(dt)

X = dt.iloc[:,0:11] #data
Y = dt.iloc[:,11:12] #class
kf = KFold(n_splits=100)

for train_index, test_index in kf.split(X):
	X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
	y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
	print("------------")
	print("X_test",X_test)

#Câu 3
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
model = GaussianNB()
model.fit(X_train, y_train)
thucte = y_test
dubao = model.predict(X_test) 
#print(thucte)
#print("-----------------\n")
#print(dubao)

#Câu 4
print("-----------------\n")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cnf_matrix_gnb = confusion_matrix(thucte,dubao)
print(cnf_matrix_gnb)
from sklearn.metrics import accuracy_score
print('accuracy = ',accuracy_score(thucte,dubao))

#Câu 5
sum = 0
nFold = KFold(n_splits=70, shuffle=True, random_state=100) 
for train_index1, test_index1 in nFold.split(X):
    X_train1, X_test1 = X.iloc[train_index1,], X.iloc[test_index1,]
    y_train1, y_test1 = Y.iloc[train_index1], Y.iloc[test_index1]
    #print(len(X_test)) # tap kiem tra co 48 phan tu
    #print(len(X_train)) # tap huan luyen co 4850  phan tu
    model = GaussianNB()
    model.fit(X_train1,y_train1)
    y_pred1 =model.predict(X_test1)
    tam=accuracy_score(y_test1,y_pred1)
    sum +=tam*100
    #print(tam)
print('Do chinh xac tong the cua 70 lan lap: ',sum/70)

#Câu 6
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y, test_size=0.3, random_state=0)
#model Decision Tree
model_Tree = DecisionTreeClassifier(criterion='gini',random_state = 100, max_depth = 6, min_samples_leaf = 10)
model_Tree.fit(X_train2, y_train2)
y_pred_Tree=model_Tree.predict(X_test2)
print('Do chinh xac cua mo hinh Decision Tree: ',accuracy_score(y_test2,y_pred_Tree)*100)
#model Bayes Naive
model_Bayes = GaussianNB()
model_Bayes.fit(X_train2,y_train2)
y_pred_Bayes =model_Bayes.predict(X_test2)
print('Do chinh xac cua mo hinh Bayes Naive: ',accuracy_score(y_test2,y_pred_Bayes)*100)


