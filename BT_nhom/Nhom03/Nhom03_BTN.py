#Nhóm 3
#Thành viên:
    #Trần Thị Huỳnh Như B1809385
    #Phạm Thị Như Mỵ B1809373
    #Phạm Thị Hồng Linh B1809365

import pandas as pd
import numpy as np
from scipy.io import arff

dataset=pd.read_csv('csv_result-messidor_features.csv')

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
dataset = shuffle(dataset)
dataset.reset_index (inplace = True, drop = True)
#print(dataset)
X=dataset.iloc[:,1:19]
y=dataset.iloc[:,19:20]
#print(X)
#print(y)
print('=============================================================================================')
print('So luong nhan 0 va nhan 1')
print(y.value_counts())
print('=============================================================================================')
kf=KFold(n_splits=40)
for train_index,test_index in kf.split(X):
	X_train,X_test=X.iloc[train_index,], X.iloc[test_index,]
	y_train,y_test=y.iloc[train_index],y.iloc[test_index]
print(X_test)
print(X_train) 
print('=============================================================================================')
#Bayes
print('BAYES')
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
model=GaussianNB()
model.fit(X_train,y_train)

thucte=y_test
dubao=model.predict(X_test)
print('Gia tri thuc te cua y_test')
print(np.array(thucte).T)

print('--------------------------------------')
print('Gia tri du bao bang bayes cua y_test')
print(dubao)

print('--------------------------------------')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
acc_sc=accuracy_score(thucte,dubao)*100
print("Do chinh xac xay dung bang Bayes accuracy = ",acc_sc)

print('=============================================================================================')
#Decision Tree
print('DECISION TREE')
from sklearn.tree import DecisionTreeClassifier
model_T=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=6,min_samples_leaf=10)
model_T.fit(X_train,y_train)

y_pred_T=model_T.predict(X_test)
print('Gia tri thuc te cua y_test')
print(np.array(thucte).T)
print('--------------------------------------')
print('Gia tri du bao bang Decision Tree cua y_test')
print(y_pred_T)

print('--------------------------------------')
twodc=X_test.iloc[-2:,:]
y_two=y_test.iloc[-2:,:]
print('Du lieu 2 dong cuoi trong X_test')
print(twodc)
print('Gia tri thuc te cua 2 dong cuoi trong y_test')
print(y_two)
print('--------------------------------------')
print('Gia tri du bao cua 2 dong cuoi bang decision tree trong y_test')
twodb=model_T.predict(twodc)
print(twodb)
print('--------------------------------------')

print('Du lieu du bao')
z=[[1,22,22,22,19,18,14,49,17,5,0.8,0.02,0.0068,0.0039,0.0039,0.5,0.1,1]]
print(z)
zdb=model_T.predict(z)
print('Ket qua du bao: ')
print(zdb)
if zdb==0:
    print('=> Khong mac benh')
else:
    print('=> Bi benh vong mac tieu duong')
print('--------------------------------------')
    
print("Do chinh xac xay dung bang Decision Tree accuracy= ",accuracy_score(y_test,y_pred_T)*100)

print('=============================================================================================')
#Do chinh xac tong the cua cac giai thuat lap 10 lan
tong = 0
tong2=0
nFold = KFold(n_splits=10, shuffle=True, random_state=100) 
for train_index1, test_index1 in nFold.split(X):
    
    X_train1, X_test1 = X.iloc[train_index1,], X.iloc[test_index1,]
    y_train1, y_test1 = y.iloc[train_index1], y.iloc[test_index1]
    model = GaussianNB()
    model.fit(X_train1,y_train1)
    y_pred1 =model.predict(X_test1)
    temp=accuracy_score(y_test1,y_pred1)
    print(temp*100)
    tong +=temp*100
    
    model_TT=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=6,min_samples_leaf=10)
    model_TT.fit(X_train1,y_train1)
    y_pred_TT=model_T.predict(X_test1)
    temp2=accuracy_score(y_test1,y_pred_TT)
    print(temp2*100)
    tong2 +=temp2*100
    
print('=============================================================================================')
print('Do chinh xac tong the cua Bayes 10 lan lap = ',tong/10)
print('Do chinh xac tong the cua Decision Tree 10 lan lap = ',tong2/10)