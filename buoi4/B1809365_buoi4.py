#Ho ten: Pham Thi Hong Linh
#MSSV: B1809365
#BAI THUC HANH 4

#Bai 1
import numpy as np
import 
 as plt
import pandas as pd

print("---------BAI 1-----------")
X = np.array([1,2,4])
Y = np.array([2,3,6])

def LR2(X, Y, eta, lanlap, theta0, theta1):
	m= len(X)
	for i in range(0,lanlap):
		print("Lan lap: ",i)
		t=0
		l=0
		for j in range(0,m):
			h=(Y[j] - (theta0+theta1*X[j]))*1
			t=t+h
			k=(Y[j] - (theta0+theta1*X[j]))*X[j]
			l=l+k
		theta0 = theta0 + eta*t
		theta1 = theta1 + eta*l
		print(theta0)
		print(theta1)
	return [round(theta0,3),round(theta1,3)]

theta = LR2(X,Y,0.2,2,0,1)
print (theta)

XX = [0,3,5]
for i in range(0,3):
	YY = theta[0] + theta[1]*XX[i]
	print (round(YY,3))

#Ham hoi quy h(x) = -1.46 - 4.96x

#Gia tri du bao y cua phan tu
    # x = 0 ; y = -1.96
    # x = 3 ; y = -16.84
    # x = 5 ; y = -26.76
    
#Ket qua cua giai thuat LR2 so voi LR1
 # LR2 yêu cầu số lần lặp cao hơn LR1, nhưng về mặt tính toán đơn giản tối ưu hơn so với LR1


#Bai2
print("---------BAI 2-----------")
dt = pd.read_csv("data2.csv", index_col=0)
dt.iloc[0:12,]
X=dt.iloc[:,[0]]
X.iloc[0:12,]
Y=dt.Y

plt.scatter(dt.X,dt.Y)
#plt.show()

import sklearn 
from sklearn import linear_model
lm = linear_model.LinearRegression()
lm.fit(X[0:12],Y[0:12])

print (lm.intercept_)
print (lm.coef_)

Y=dt.Y
y_test = Y[-2:]
X_test = X[-2:]
y_pred = lm.predict(X_test)

print(y_pred)
print(y_test)

from sklearn.metrics import mean_squared_error
err = mean_squared_error(y_test, y_pred)
print(err)
rmse_err = np.sqrt(err)
round(rmse_err,3)

#Gia tri Y thuc te 
    #phan tu 13 : 10
	#phan tu 14 : 10
#Gia tri Y du doan  bởi Sklearn
    #phan tu 13 : 9.75826542
	#phan tu 14 : 10.39724478

