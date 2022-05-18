#Bieu dien du lieu len mptd
#a
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,4])
Y = np.array([2,3,6])

plt.axis([0,5,0,8])
plt.plot(X,Y,"ro",color="blue")
plt.xlabel("Gia tri thuoc tinh X")
plt.ylabel("Gia tri du doan Y")
#plt.show()
#b
def LR1(X,Y,eta,lanlap, theta0,theta1):
	m = len(X)
	for i in range(0,lanlap):
		print("Lan lap: ",i)
		for j in range(0,m):
			#theta0
			h= theta0 + theta1*X[j]
			theta0 = theta0 + eta*(Y[j]-h)*1
			print("Phan tu ", j, "y=", Y[j], "h=", h,"gia tri theta0 = ",theta0)
			#theta1
			theta1 = theta1 + eta*(Y[j]-h)*X[j]
			print("Phan tu ", j,"gia tri theta1 = ", theta1)
	return [theta0,theta1]
	
theta = LR1(X,Y,0.2,1,0,1)
print(theta)
theta2 = LR1(X,Y,0.2,2,0,1)
print(theta2)

XX = [0,3,5]
for i in range(0,3):
	YY = theta[0] + theta[1]*XX[i]
	print (round(YY,3))
 #0.336 5.088 8.256

#c ve duong hoi quy
theta = LR1(X,Y,0.1,1,0,1)
X1= np.array([1,6])
Y1= theta[0] + theta[1]*X1

theta2 = LR1(X,Y,0.1,2,0,1)
X2= np.array([1,6])
Y2= theta2[0] + theta2[1]*X2

plt.axis([0,7,0,10])
plt.plot(X,Y,"ro",color="blue")

plt.plot(X1,Y1,color="violet")
plt.plot(X2,Y2,color="green")

plt.xlabel("Gia tri thuoc tinh X")
plt.ylabel("Gia tri du doan Y")
#plt.show()

#e 
y1= theta[0] + theta[1]*0
y2= theta[0] + theta[1]*3
y3= theta[0] + theta[1]*5
print(y1)
print(y2)
print(y3)
