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
print(X)
print(y)

kf=KFold(n_splits=40)
for train_index,test_index in kf.split(X):
	X_train,X_test=X.iloc[train_index,], X.iloc[test_index,]
	y_train,y_test=y.iloc[train_index],y.iloc[test_index]
print(X_test)
print(X_train) 

#Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
model=GaussianNB()
model.fit(X_train,y_train)

thucte=y_test
dubao=model.predict(X_test)
print('Gia tri thuc te')
print(np.array(thucte).T)
print('--------------------------------------')
print('Gia tri du bao bang bayes')
print(dubao)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
acc_sc=accuracy_score(thucte,dubao)*100
print("Do chinh xac xay dung bang Bayes accuracy = ",acc_sc)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_T=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=6,min_samples_leaf=10)
model_T.fit(X_train,y_train)

y_pred_T=model_T.predict(X_test)
print('Gia tri thuc te')
print(np.array(thucte).T)
print('--------------------------------------')
print('Gia tri du bao bang Decision Tree')
print(y_pred_T)
print("Do chinh xac xay dung bang Decision Tree accuracy= ",accuracy_score(y_test,y_pred_T)*100)

#Do chinh xac tong the cua cac giai thuat lap 70 lan
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
    

print('Do chinh xac tong the cua Bayes 10 lan lap = ',tong/10)
print('Do chinh xac tong the cua Decision Tree 10 lan lap = ',tong2/10)